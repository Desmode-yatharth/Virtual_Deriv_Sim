

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import heapq
from collections import defaultdict
import random
import time

def black_scholes(S, K, T, r, sigma, option_type):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 1.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        return max(price, 1.0)
    except (OverflowError, ZeroDivisionError):
        return 1.0

def future_price(S, r, sigma, T):
    if S <= 0 or T <= 0 or sigma < 0:
        return 1.0
    try:
        price = S * np.exp((r - 0.5 * sigma ** 2) * T)
        return max(price, 1.0)
    except (OverflowError, ZeroDivisionError):
        return 1.0

def simulate_gbm(S0, mu, sigma, T, dt, steps):
    t = np.linspace(0, T, steps)
    W = np.random.standard_normal(size=steps)
    W = np.cumsum(W) * np.sqrt(dt)
    return S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)



class Derivative:
    def __init__(self, underlying_price, r, sigma, T, num_options, num_futures, num_strikes, total_qty_per_deriv=10):
        self.underlying_price = underlying_price
        self.r = r
        self.sigma = sigma
        self.T = T
        self.num_options = min(num_options, 30)  # Cap at 30 calls and 30 puts
        self.num_futures = min(num_futures, 15)  # Cap at 15 futures
        self.num_strikes = num_strikes
        self.total_qty_per_deriv = total_qty_per_deriv
        self.derivs = {}
        self.init_derivs()
        self.calculate_initial_prices()

    def init_derivs(self):
        # Generate strikes with increments of 5, centered around underlying_price
        half_strikes = self.num_strikes // 2
        strike_range = [self.underlying_price + i * 5 for i in range(-half_strikes, half_strikes + 1)]
        strike_range = strike_range[:self.num_strikes]  # Limit to num_strikes
        
        # Create calls and puts (max 30 each)
        for strike in strike_range[:min(self.num_options, 30)]:
            self.create_deriv(strike, "call")
            self.create_deriv(strike, "put")
        
        # Create futures (max 15)
        for strike in strike_range[:min(self.num_futures, 15)]:
            self.create_deriv(strike, None)

    def create_deriv(self, strike_price, option_type=None):
        name = self.generate_name(strike_price, option_type)
        if name not in self.derivs:
            self.derivs[name] = (strike_price, option_type, None, self.total_qty_per_deriv)
        return name

    def generate_name(self, strike_price, option_type):
        underlying_names = {0: "ALPHA", 1: "BETA", 2: "GAMMA", 3: "DELTA"}
        name_idx = hash(str(strike_price)) % len(underlying_names)
        underlying_name = underlying_names[name_idx]
        if option_type == "call":
            return f"{underlying_name} {strike_price:.2f} Call"
        elif option_type == "put":
            return f"{underlying_name} {strike_price:.2f} Put"
        else:
            return f"{underlying_name} {strike_price:.2f} Future"

    def calculate_initial_prices(self):
        for name in self.derivs:
            strike, opt_type, _, qty = self.derivs[name]
            if opt_type in ["call", "put"]:
                price = black_scholes(self.underlying_price, strike, self.T, self.r, self.sigma, opt_type)
            else:
                price = future_price(self.underlying_price, self.r, self.sigma, self.T)
            self.derivs[name] = (strike, opt_type, price, qty)

    def get_price(self, name):
        if name in self.derivs:
            return self.derivs[name][2]
        return None

    def update_price(self, name, new_underlying_price):
        if name in self.derivs:
            strike, opt_type, _, qty = self.derivs[name]
            if opt_type in ["call", "put"]:
                price = black_scholes(new_underlying_price, strike, self.T, self.r, self.sigma, opt_type)
            else:
                price = future_price(new_underlying_price, self.r, self.sigma, self.T)
            self.derivs[name] = (strike, opt_type, price, qty)
            return price
        return None

    def update_prices(self, new_underlying_price):
        self.underlying_price = new_underlying_price
        for name in self.derivs:
            self.update_price(name, new_underlying_price)

    def get_all_derivs(self):
        return sorted(list(self.derivs.keys()))

    def get_total_qty(self, name):
        if name in self.derivs:
            return self.derivs[name][3]
        return 0

    def get_market_cap(self):
        return sum(self.get_price(name) * self.get_total_qty(name) for name in self.derivs if self.get_price(name) is not None)

    def get_lazy_price_history(self, name, price_path, current_step):
        if name not in self.derivs:
            return []
        strike, opt_type, _, _ = self.derivs[name]
        history = []
        for step in range(current_step + 1):
            underlying = price_path[step]
            if opt_type in ["call", "put"]:
                price = black_scholes(underlying, strike, self.T, self.r, self.sigma, opt_type)
            else:
                price = future_price(underlying, self.r, self.sigma, self.T)
            history.append(price)
        return history





class OrderBook:
    def __init__(self, deriv_hub):
        self.bids = []  # Heap of (price, order_id), negated for max-heap
        self.asks = []  # Heap of (price, order_id)
        self.orders = defaultdict(list)  # price: [(qty, trader, side, deriv, order_id, step_created)]
        self.order_id = 0
        self.trades = []
        self.deriv_hub = deriv_hub
        self.current_step = 0
        self.expiration_steps = 7

    def add_order(self, price, quantity, trader, side, deriv):
        # Check for existing active order at ANY price for same trader and deriv
        for price_level in self.orders:
            for order in self.orders[price_level]:
                if order[1] == trader and order[3] == deriv and (self.current_step - order[5] < self.expiration_steps):
                    st.session_state.trade_log.append(f"Order Rejected: {trader.name} already has active order for {deriv}")
                    return
        self.order_id += 1
        order = (quantity, trader, side, deriv, self.order_id, self.current_step)
        if side == "buy":
            heapq.heappush(self.bids, (-price, self.order_id))  # Negate price for max-heap
            self.orders[price].append(order)
            st.session_state.trade_log.append(f"Order Added: {trader.name} bids {quantity} {deriv} at ${price:.2f}")
        else:
            heapq.heappush(self.asks, (price, self.order_id))
            self.orders[price].append(order)
            st.session_state.trade_log.append(f"Order Added: {trader.name} asks {quantity} {deriv} at ${price:.2f}")
        return self.order_id

    def expire_orders(self):
        expired = False
        for price in list(self.orders.keys()):
            orders_at_price = self.orders[price]
            for order in orders_at_price[:]:
                qty, trader, side, deriv, order_id, step_created = order
                if self.current_step - step_created >= self.expiration_steps:
                    orders_at_price.remove(order)
                    if side == "buy":
                        self.bids = [(p, oid) for p, oid in self.bids if oid != order_id]
                        heapq.heapify(self.bids)
                    else:
                        self.asks = [(p, oid) for p, oid in self.asks if oid != order_id]
                        heapq.heapify(self.asks)
                    expired = True
                    st.session_state.expiration_log.append(f"Expired: {trader.name} {side} {qty} {deriv} at ${price:.2f} (Order ID: {order_id})")
                    if trader.name == "User1":
                        trader.expire_user_orders(self.current_step)
            if not orders_at_price:
                del self.orders[price]
        return expired

    def match_orders(self):
        self.expire_orders()

        def trader_priority(trader, is_counterparty=False, user_step=None):
            if trader.name == "User1":
                return 0  # Highest priority for User1
            elif trader.name.startswith("LB"):
                return 1 if not is_counterparty or user_step is None else 1  # LBs priority for User1 same-step
            else:
                return 2 if not is_counterparty or user_step is None else 3  # Normal bots lower

        while self.bids and self.asks:
            best_bid_price = -self.bids[0][0] if self.bids else float('-inf')  # Negated back to positive
            best_ask_price = self.asks[0][0] if self.asks else float('inf')
            if best_bid_price < best_ask_price:  # No overlap, stop matching
                break

            bid_id = self.bids[0][1]
            ask_id = self.asks[0][1]
            bid_orders = self.orders[best_bid_price]
            ask_orders = self.orders[best_ask_price]

            bid = next((o for o in bid_orders if o[4] == bid_id), None)
            ask = next((o for o in ask_orders if o[4] == ask_id), None)
            if not bid or not ask:
                st.session_state.trade_log.append(f"Match Failed: Bid ID {bid_id} or Ask ID {ask_id} missing")
                if not bid:
                    heapq.heappop(self.bids)
                if not ask:
                    heapq.heappop(self.asks)
                continue

            bid_qty, bid_trader, _, bid_deriv, bid_order_id, bid_step = bid
            ask_qty, ask_trader, _, ask_deriv, ask_order_id, ask_step = ask

            if bid_deriv != ask_deriv or bid_trader == ask_trader:
                st.session_state.trade_log.append(f"Match Skipped: {bid_deriv} vs {ask_deriv} or same trader")
                heapq.heappop(self.bids)
                heapq.heappop(self.asks)
                bid_orders.remove(bid)
                ask_orders.remove(ask)
                if not bid_orders:
                    del self.orders[best_bid_price]
                if not ask_orders:
                    del self.orders[best_ask_price]
                continue

            # User1 same-step priority for LB counterparties
            user_step = None
            if bid_trader.name == "User1" and bid_step == self.current_step:
                user_step = bid_step
                ask_orders.sort(key=lambda x: trader_priority(x[1], is_counterparty=True, user_step=user_step))
                ask = ask_orders[0]  # Re-pick ask
                ask_qty, ask_trader, _, ask_deriv, ask_order_id, ask_step = ask
            elif ask_trader.name == "User1" and ask_step == self.current_step:
                user_step = ask_step
                bid_orders.sort(key=lambda x: trader_priority(x[1], is_counterparty=True, user_step=user_step))
                bid = bid_orders[0]  # Re-pick bid
                bid_qty, bid_trader, _, bid_deriv, bid_order_id, bid_step = bid

            matched_qty = min(bid_qty, ask_qty)
            trade_price = (best_bid_price + best_ask_price) / 2

            if bid_trader.cash >= trade_price * matched_qty:
                bid_trader.cash -= trade_price * matched_qty
                ask_trader.cash += trade_price * matched_qty
                bid_trader.positions[bid_deriv] = bid_trader.positions.get(bid_deriv, 0) + matched_qty
                ask_trader.positions[ask_deriv] = ask_trader.positions.get(ask_deriv, 0) - matched_qty
                self.trades.append({
                    "Buyer": bid_trader.name,
                    "Seller": ask_trader.name,
                    "Derivative": bid_deriv,
                    "Quantity": matched_qty,
                    "Price": trade_price
                })
                st.session_state.trade_log.append(f"Trade: {bid_trader.name} buys {matched_qty} {bid_deriv} from {ask_trader.name} at ${trade_price:.2f}")

                if bid_trader.name == "User1":
                    for u_order in bid_trader.user_orders[:]:
                        if u_order["Derivative"] == bid_deriv and u_order["Status"] == "Pending" and u_order["OrderID"] == bid_order_id:
                            if bid_qty == matched_qty:
                                bid_trader.user_orders.remove(u_order)
                            else:
                                u_order["Quantity"] -= matched_qty
                            break
                if ask_trader.name == "User1":
                    for u_order in ask_trader.user_orders[:]:
                        if u_order["Derivative"] == ask_deriv and u_order["Status"] == "Pending" and u_order["OrderID"] == ask_order_id:
                            if ask_qty == matched_qty:
                                ask_trader.user_orders.remove(u_order)
                            else:
                                u_order["Quantity"] -= matched_qty
                            break

                if bid_qty > matched_qty:
                    bid_orders[bid_orders.index(bid)] = (bid_qty - matched_qty, bid_trader, "buy", bid_deriv, bid_order_id, bid_step)
                else:
                    bid_orders.remove(bid)
                    heapq.heappop(self.bids)
                if ask_qty > matched_qty:
                    ask_orders[ask_orders.index(ask)] = (ask_qty - matched_qty, ask_trader, "sell", ask_deriv, ask_order_id, ask_step)
                else:
                    ask_orders.remove(ask)
                    heapq.heappop(self.asks)

                if not bid_orders:
                    del self.orders[best_bid_price]
                if not ask_orders:
                    del self.orders[best_ask_price]
            else:
                st.session_state.trade_log.append(f"Order Failed: {bid_trader.name} insufficient cash (${bid_trader.cash:.2f})")
                heapq.heappop(self.bids)
                bid_orders.remove(bid)
                if not bid_orders:
                    del self.orders[best_bid_price]
                break

    def display(self):
        bids_data = []
        asks_data = []
        for price in self.orders:
            for qty, trader, side, deriv, _, _ in self.orders[price]:
                if side == "buy":
                    bids_data.append({"Price": price, "Quantity": qty, "Trader": trader.name, "Derivative": deriv})
                else:
                    asks_data.append({"Price": price, "Quantity": qty, "Trader": trader.name, "Derivative": deriv})
        return {"Bids": pd.DataFrame(bids_data), "Asks": pd.DataFrame(asks_data)}

    def get_trades(self):
        return pd.DataFrame(self.trades)

    def reset(self):
        self.bids = []
        self.asks = []
        self.orders = defaultdict(list)
        self.order_id = 0
        self.trades = []
        self.current_step = 0

class UserTrader:
    def __init__(self, name, cash=10000, deriv_hub=None, pre_owned_n=0):
        self.name = name
        self.cash = cash
        self.positions = {}
        self.deriv_hub = deriv_hub
        self.selected_deriv = None
        self.pre_owned_n = pre_owned_n
        self.user_orders = []
        self.init_pre_owned()

    def init_pre_owned(self):
        if self.pre_owned_n == 0 or not self.deriv_hub:
            return
        all_derivs = self.deriv_hub.get_all_derivs()
        calls = [d for d in all_derivs if "Call" in d]
        puts = [d for d in all_derivs if "Put" in d]
        futures = [d for d in all_derivs if "Future" in d]
        total_qty_per_deriv = self.deriv_hub.total_qty_per_deriv
        qty_per_type = min(self.pre_owned_n, total_qty_per_deriv)
        for derivs_list in [calls, puts, futures]:
            selected = random.sample(derivs_list, min(len(derivs_list), self.pre_owned_n))
            for deriv in selected:
                self.positions[deriv] = qty_per_type
        st.session_state.init_log.append(f"{self.name} initialized with {len(self.positions)} derivs, {qty_per_type} each")

    def place_order(self, order_book, deriv, price, quantity, side):
        adjusted_price = price * (1 + random.uniform(-0.02, 0.02))
        order_details = {
            "Side": side,
            "Derivative": deriv,
            "Quantity": quantity,
            "Price": adjusted_price,
            "Status": "Pending",
            "Steps Remaining": 7,
            "Step Placed": order_book.current_step
        }
        if side == "buy" and self.cash >= adjusted_price * quantity:
            order_book.add_order(adjusted_price, quantity, self, side, deriv)
            self.user_orders.append(order_details)
            st.session_state.trade_log.append(f"User Buy Order: {quantity} {deriv} at ${adjusted_price:.2f}")
        elif side == "sell" and self.positions.get(deriv, 0) >= quantity:
            order_book.add_order(adjusted_price, quantity, self, side, deriv)
            self.user_orders.append(order_details)
            st.session_state.trade_log.append(f"User Sell Order: {quantity} {deriv} at ${adjusted_price:.2f}")
        else:
            order_details["Status"] = "Failed"
            self.user_orders.append(order_details)
            st.session_state.trade_log.append(f"User Order Failed: {side} {quantity} {deriv} - Cash: ${self.cash:.2f}, Pos: {self.positions.get(deriv, 0)}")

    def expire_user_orders(self, current_step):
        for order in self.user_orders[:]:
            if order["Status"] == "Pending":
                steps_elapsed = current_step - order["Step Placed"]
                order["Steps Remaining"] = max(0, 7 - steps_elapsed)
                if order["Steps Remaining"] == 0:
                    order["Status"] = "Expired"
                    st.session_state.expiration_log.append(f"Expired: {self.name} {order['Side']} {order['Quantity']} {order['Derivative']} at ${order['Price']:.2f}")

    def reset(self, new_cash=10000):
        self.cash = new_cash
        self.positions = {}
        self.selected_deriv = None
        self.user_orders = []
        self.init_pre_owned()






class VirtualTrader:
    def __init__(self, name, cash=10000, deriv_hub=None, all_traders=None, pre_owned_n=0):
        self.name = name
        self.cash = cash
        self.positions = {}
        self.deriv_hub = deriv_hub
        self.all_traders = all_traders
        self.pre_owned_n = pre_owned_n
        self.available_derivs = self.deriv_hub.get_all_derivs() if deriv_hub else []
        self.trade_counter = 0
        self.active_orders = {}
        self.max_short = -10  # Cap on short positions
        self.short_sell_prob = 0.1  # 10% chance to short-sell

    def init_pre_owned(self):
        if self.pre_owned_n == 0 or not self.deriv_hub:
            return
        all_derivs = self.deriv_hub.get_all_derivs()
        calls = [d for d in all_derivs if "Call" in d]
        puts = [d for d in all_derivs if "Put" in d]
        futures = [d for d in all_derivs if "Future" in d]
        total_qty_per_deriv = self.deriv_hub.total_qty_per_deriv
        qty_per_type = min(self.pre_owned_n, total_qty_per_deriv)
        for derivs_list in [calls, puts, futures]:
            selected = random.sample(derivs_list, min(len(derivs_list), self.pre_owned_n))
            for deriv in selected:
                self.positions[deriv] = qty_per_type
        st.session_state.init_log = st.session_state.get('init_log', []) + [f"{self.name} initialized with {len(self.positions)} derivs, {qty_per_type} each"]

    def simulate_trade(self, order_book, user_trader, bots, mode):
        self.trade_counter += 1
        trade_frequency = {"Passive": 4, "Neutral": 3, "Aggro": 2, "Very Aggro": 1}[mode]
        owned_derivs = [d for d in self.positions.keys() if self.positions[d] > 0]

        for deriv in list(self.active_orders.keys()):
            order_id, step_placed, _ = self.active_orders[deriv]
            order_exists = any(o[4] == order_id for price in order_book.orders for o in order_book.orders[price])
            if not order_exists or (order_book.current_step - step_placed >= order_book.expiration_steps):
                del self.active_orders[deriv]

        if self.trade_counter % trade_frequency == 0:
            self.trade_counter = 0
            derivs_to_trade = [d for d in (owned_derivs or self.available_derivs) if d not in self.active_orders]
            if not derivs_to_trade:
                return
            deriv_name = random.choice(derivs_to_trade)
            current_price = self.deriv_hub.get_price(deriv_name)
            if current_price is None:
                return
            qty = random.randint(1, 5)
            price = current_price * (1 + random.uniform(-0.01, 0.01))
            side = random.choice(["buy", "sell"])
            current_qty = self.positions.get(deriv_name, 0)

            if side == "buy" and self.cash >= price * qty:
                order_book.add_order(price, qty, self, "buy", deriv_name)
                self.active_orders[deriv_name] = (order_book.order_id, order_book.current_step, "buy")
            elif side == "sell":
                if current_qty >= qty:  # Normal trade: sell owned
                    order_book.add_order(price, qty, self, "sell", deriv_name)
                    self.active_orders[deriv_name] = (order_book.order_id, order_book.current_step, "sell")
                elif current_qty > self.max_short and random.random() < self.short_sell_prob:  # Rare short-sell
                    order_book.add_order(price, qty, self, "sell", deriv_name)
                    self.active_orders[deriv_name] = (order_book.order_id, order_book.current_step, "sell")

    def reset(self, new_cash=10000):
        self.cash = new_cash
        self.positions = {}
        self.available_derivs = self.deriv_hub.get_all_derivs() if self.deriv_hub else []
        self.trade_counter = 0
        self.active_orders = {}





class LiquidityBotCalls:
    def __init__(self, name, cash=10000, deriv_hub=None):
        self.name = name
        self.cash = cash
        self.positions = {}
        self.deriv_hub = deriv_hub
        self.focus_type = "Call"
        self.target_pct = 0.35
        self.trade_counter = 0
        self.active_orders = {}
        self.max_short = -50
        self.short_sell_prob = 0.1
        self.flood_steps = 3

    def assign_remaining(self, all_traders, derivs):
        calls = [d for d in derivs if "Call" in d]
        if not calls:
            st.session_state.trade_log.append(f"{self.name}: No Calls available to assign")
        for deriv in calls:
            total_owned = sum(t.positions.get(deriv, 0) for t in all_traders)
            remaining = self.deriv_hub.total_qty_per_deriv - total_owned
            if remaining > 0:
                self.positions[deriv] = remaining
                st.session_state.init_log = st.session_state.get('init_log', []) + [f"{self.name} takes {remaining} {deriv}"]

    def initial_auction(self, order_book):
        calls = [d for d in self.deriv_hub.get_all_derivs() if "Call" in d]
        if not calls:
            st.session_state.trade_log.append(f"{self.name}: No Calls found for auction")
            return
        num_derivs_to_trade = min(max(5, len(calls)), 10)
        selected_derivs = random.sample(calls, min(num_derivs_to_trade, len(calls)))
        for deriv in selected_derivs:
            if deriv not in self.active_orders:
                current_qty = self.positions.get(deriv, 0)
                if current_qty > 0:
                    price = self.deriv_hub.get_price(deriv)
                    if price is None:
                        st.session_state.trade_log.append(f"{self.name}: No price for {deriv}")
                        continue
                    price *= 1.02
                    qty = min(int(self.deriv_hub.total_qty_per_deriv * random.uniform(0.75, 1.0)), current_qty)
                    order_book.add_order(price, qty, self, "sell", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
                    st.session_state.trade_log.append(f"Auction: {self.name} sells {qty} {deriv} at ${price:.2f}")

    def force_trade(self, order_book, mode):
        self.trade_counter += 1
        trade_frequency = {"Passive": 4, "Neutral": 3, "Aggro": 2, "Very Aggro": 1}[mode]
        calls = [d for d in self.deriv_hub.get_all_derivs() if "Call" in d]
        if not calls:
            st.session_state.trade_log.append(f"{self.name}: No Calls found for trading")
            return
        target_qty = int(self.deriv_hub.total_qty_per_deriv * self.target_pct)

        for deriv in list(self.active_orders.keys()):
            order_id, step_placed, _ = self.active_orders[deriv]
            order_exists = any(o[4] == order_id for price in order_book.orders for o in order_book.orders[price])
            if not order_exists or (order_book.current_step - step_placed >= order_book.expiration_steps):
                del self.active_orders[deriv]
                st.session_state.trade_log.append(f"{self.name}: Order for {deriv} removed (expired or matched)")

        available_calls = [d for d in calls if d not in self.active_orders]
        if not available_calls:
            st.session_state.trade_log.append(f"{self.name}: No available Calls to trade")
            return

        for _ in range(min(3, len(available_calls))):
            deriv = random.choice(available_calls)
            current_qty = self.positions.get(deriv, 0)
            current_price = self.deriv_hub.get_price(deriv)
            if current_price is None:
                st.session_state.trade_log.append(f"{self.name}: No price for {deriv}")
                continue

            if 1 <= order_book.current_step <= self.flood_steps:
                if current_qty > 0:
                    qty = min(int(self.deriv_hub.total_qty_per_deriv * random.uniform(0.5, 0.75)), current_qty)
                    order_book.add_order(current_price * 1.02, qty, self, "sell", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
            elif order_book.current_step >= 4 and self.trade_counter >= trade_frequency:
                self.trade_counter = 0
                qty = random.randint(1, 5)
                if current_qty < target_qty and self.cash >= current_price * 0.98 * qty:
                    order_book.add_order(current_price * 0.98, qty, self, "buy", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "buy")
                elif current_qty >= target_qty and current_qty > 0:
                    sell_qty = min(qty, current_qty - target_qty)
                    if sell_qty > 0:
                        order_book.add_order(current_price * 1.02, sell_qty, self, "sell", deriv)
                        self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
                elif current_qty > self.max_short and random.random() < self.short_sell_prob:
                    order_book.add_order(current_price * 1.02, qty, self, "sell", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
            available_calls.remove(deriv)

    def reset(self, new_cash=10000):
        self.cash = new_cash
        self.positions = {}
        self.trade_counter = 0
        self.active_orders = {}

class LiquidityBotPuts:
    def __init__(self, name, cash=10000, deriv_hub=None):
        self.name = name
        self.cash = cash
        self.positions = {}
        self.deriv_hub = deriv_hub
        self.focus_type = "Put"
        self.target_pct = 0.35
        self.trade_counter = 0
        self.active_orders = {}
        self.max_short = -50
        self.short_sell_prob = 0.1
        self.flood_steps = 3

    def assign_remaining(self, all_traders, derivs):
        puts = [d for d in derivs if "Put" in d]
        if not puts:
            st.session_state.trade_log.append(f"{self.name}: No Puts available to assign")
        for deriv in puts:
            total_owned = sum(t.positions.get(deriv, 0) for t in all_traders)
            remaining = self.deriv_hub.total_qty_per_deriv - total_owned
            if remaining > 0:
                self.positions[deriv] = remaining
                st.session_state.init_log = st.session_state.get('init_log', []) + [f"{self.name} takes {remaining} {deriv}"]

    def initial_auction(self, order_book):
        puts = [d for d in self.deriv_hub.get_all_derivs() if "Put" in d]
        if not puts:
            st.session_state.trade_log.append(f"{self.name}: No Puts found for auction")
            return
        num_derivs_to_trade = min(max(5, len(puts)), 10)
        selected_derivs = random.sample(puts, min(num_derivs_to_trade, len(puts)))
        for deriv in selected_derivs:
            if deriv not in self.active_orders:
                current_qty = self.positions.get(deriv, 0)
                if current_qty > 0:
                    price = self.deriv_hub.get_price(deriv)
                    if price is None:
                        st.session_state.trade_log.append(f"{self.name}: No price for {deriv}")
                        continue
                    price *= 1.02
                    qty = min(int(self.deriv_hub.total_qty_per_deriv * random.uniform(0.75, 1.0)), current_qty)
                    order_book.add_order(price, qty, self, "sell", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
                    st.session_state.trade_log.append(f"Auction: {self.name} sells {qty} {deriv} at ${price:.2f}")

    def force_trade(self, order_book, mode):
        self.trade_counter += 1
        trade_frequency = {"Passive": 4, "Neutral": 3, "Aggro": 2, "Very Aggro": 1}[mode]
        puts = [d for d in self.deriv_hub.get_all_derivs() if "Put" in d]
        if not puts:
            st.session_state.trade_log.append(f"{self.name}: No Puts found for trading")
            return
        target_qty = int(self.deriv_hub.total_qty_per_deriv * self.target_pct)

        for deriv in list(self.active_orders.keys()):
            order_id, step_placed, _ = self.active_orders[deriv]
            order_exists = any(o[4] == order_id for price in order_book.orders for o in order_book.orders[price])
            if not order_exists or (order_book.current_step - step_placed >= order_book.expiration_steps):
                del self.active_orders[deriv]
                st.session_state.trade_log.append(f"{self.name}: Order for {deriv} removed (expired or matched)")

        available_puts = [d for d in puts if d not in self.active_orders]
        if not available_puts:
            st.session_state.trade_log.append(f"{self.name}: No available Puts to trade")
            return

        for _ in range(min(3, len(available_puts))):
            deriv = random.choice(available_puts)
            current_qty = self.positions.get(deriv, 0)
            current_price = self.deriv_hub.get_price(deriv)
            if current_price is None:
                st.session_state.trade_log.append(f"{self.name}: No price for {deriv}")
                continue

            if 1 <= order_book.current_step <= self.flood_steps:
                if current_qty > 0:
                    qty = min(int(self.deriv_hub.total_qty_per_deriv * random.uniform(0.5, 0.75)), current_qty)
                    order_book.add_order(current_price * 1.02, qty, self, "sell", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
            elif order_book.current_step >= 4 and self.trade_counter >= trade_frequency:
                self.trade_counter = 0
                qty = random.randint(1, 5)
                if current_qty < target_qty and self.cash >= current_price * 0.98 * qty:
                    order_book.add_order(current_price * 0.98, qty, self, "buy", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "buy")
                elif current_qty >= target_qty and current_qty > 0:
                    sell_qty = min(qty, current_qty - target_qty)
                    if sell_qty > 0:
                        order_book.add_order(current_price * 1.02, sell_qty, self, "sell", deriv)
                        self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
                elif current_qty > self.max_short and random.random() < self.short_sell_prob:
                    order_book.add_order(current_price * 1.02, qty, self, "sell", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
            available_puts.remove(deriv)

    def reset(self, new_cash=10000):
        self.cash = new_cash
        self.positions = {}
        self.trade_counter = 0
        self.active_orders = {}

class LiquidityBotFutures:
    def __init__(self, name, cash=10000, deriv_hub=None):
        self.name = name
        self.cash = cash
        self.positions = {}
        self.deriv_hub = deriv_hub
        self.focus_type = "Future"
        self.target_pct = 0.35
        self.trade_counter = 0
        self.active_orders = {}
        self.max_short = -50
        self.short_sell_prob = 0.1
        self.flood_steps = 3

    def assign_remaining(self, all_traders, derivs):
        futures = [d for d in derivs if "Future" in d]
        if not futures:
            st.session_state.trade_log.append(f"{self.name}: No Futures available to assign")
        for deriv in futures:
            total_owned = sum(t.positions.get(deriv, 0) for t in all_traders)
            remaining = self.deriv_hub.total_qty_per_deriv - total_owned
            if remaining > 0:
                self.positions[deriv] = remaining
                st.session_state.init_log = st.session_state.get('init_log', []) + [f"{self.name} takes {remaining} {deriv}"]

    def initial_auction(self, order_book):
        futures = [d for d in self.deriv_hub.get_all_derivs() if "Future" in d]
        if not futures:
            st.session_state.trade_log.append(f"{self.name}: No Futures found for auction")
            return
        num_derivs_to_trade = min(max(5, len(futures)), 10)
        selected_derivs = random.sample(futures, min(num_derivs_to_trade, len(futures)))
        for deriv in selected_derivs:
            if deriv not in self.active_orders:
                current_qty = self.positions.get(deriv, 0)
                if current_qty > 0:
                    price = self.deriv_hub.get_price(deriv)
                    if price is None:
                        st.session_state.trade_log.append(f"{self.name}: No price for {deriv}")
                        continue
                    price *= 1.02
                    qty = min(int(self.deriv_hub.total_qty_per_deriv * random.uniform(0.75, 1.0)), current_qty)
                    order_book.add_order(price, qty, self, "sell", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
                    st.session_state.trade_log.append(f"Auction: {self.name} sells {qty} {deriv} at ${price:.2f}")

    def force_trade(self, order_book, mode):
        self.trade_counter += 1
        trade_frequency = {"Passive": 4, "Neutral": 3, "Aggro": 2, "Very Aggro": 1}[mode]
        futures = [d for d in self.deriv_hub.get_all_derivs() if "Future" in d]
        if not futures:
            st.session_state.trade_log.append(f"{self.name}: No Futures found for trading")
            return
        target_qty = int(self.deriv_hub.total_qty_per_deriv * self.target_pct)

        for deriv in list(self.active_orders.keys()):
            order_id, step_placed, _ = self.active_orders[deriv]
            order_exists = any(o[4] == order_id for price in order_book.orders for o in order_book.orders[price])
            if not order_exists or (order_book.current_step - step_placed >= order_book.expiration_steps):
                del self.active_orders[deriv]
                st.session_state.trade_log.append(f"{self.name}: Order for {deriv} removed (expired or matched)")

        available_futures = [d for d in futures if d not in self.active_orders]
        if not available_futures:
            st.session_state.trade_log.append(f"{self.name}: No available Futures to trade")
            return

        for _ in range(min(3, len(available_futures))):
            deriv = random.choice(available_futures)
            current_qty = self.positions.get(deriv, 0)
            current_price = self.deriv_hub.get_price(deriv)
            if current_price is None:
                st.session_state.trade_log.append(f"{self.name}: No price for {deriv}")
                continue

            if 1 <= order_book.current_step <= self.flood_steps:
                if current_qty > 0:
                    qty = min(int(self.deriv_hub.total_qty_per_deriv * random.uniform(0.5, 0.75)), current_qty)
                    order_book.add_order(current_price * 1.02, qty, self, "sell", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
            elif order_book.current_step >= 4 and self.trade_counter >= trade_frequency:
                self.trade_counter = 0
                qty = random.randint(1, 5)
                if current_qty < target_qty and self.cash >= current_price * 0.98 * qty:
                    order_book.add_order(current_price * 0.98, qty, self, "buy", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "buy")
                elif current_qty >= target_qty and current_qty > 0:
                    sell_qty = min(qty, current_qty - target_qty)
                    if sell_qty > 0:
                        order_book.add_order(current_price * 1.02, sell_qty, self, "sell", deriv)
                        self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
                elif current_qty > self.max_short and random.random() < self.short_sell_prob:
                    order_book.add_order(current_price * 1.02, qty, self, "sell", deriv)
                    self.active_orders[deriv] = (order_book.order_id, order_book.current_step, "sell")
            available_futures.remove(deriv)

    def reset(self, new_cash=10000):
        self.cash = new_cash
        self.positions = {}
        self.trade_counter = 0
        self.active_orders = {}


def main():
    st.title("Virtual Derivative Market Simulator")

    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'running_auto' not in st.session_state:
        st.session_state.running_auto = False
    if 'auto_step' not in st.session_state:
        st.session_state.auto_step = 0
    if 'display_all_derivs' not in st.session_state:
        st.session_state.display_all_derivs = False
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    if 'init_log' not in st.session_state:
        st.session_state.init_log = []
    if 'expiration_log' not in st.session_state:
        st.session_state.expiration_log = []
    if 'pending_user_order' not in st.session_state:
        st.session_state.pending_user_order = None
    if 'executed_orders' not in st.session_state:
        st.session_state.executed_orders = set()

    # Sidebar parameters with updated GBM ranges
    with st.sidebar:
        st.header("Simulation Parameters")
        S0 = st.slider("Initial Asset Price (S0)", 50.0, 1000.0, 100.0, step=10.0, key="s0")
        mu = st.slider("Expected Return (mu)", 0.0, 1.0, 0.05, step=0.01, key="mu")
        sigma = st.slider("Volatility (sigma)", 0.1, 2.0, 0.2, step=0.05, key="sigma")
        r = st.slider("Risk-Free Rate (r)", 0.0, 0.5, 0.03, step=0.01, key="r")
        T = st.slider("Time Horizon (T, years)", 0.5, 10.0, 1.0, step=0.1, key="t")
        steps = st.slider("Simulation Steps", 10, 500, 25, step=5, key="steps")
        num_virtual_traders = st.slider("Number of Virtual Traders", 1, 50, 5, key="traders")
        num_options = st.slider("Number of Options (Calls = Puts)", 5, 30, 10, step=5, key="num_options")
        num_futures = st.slider("Number of Futures", 5, 15, 10, step=5, key="num_futures")
        num_strikes = st.slider("Number of Strike Prices", 5, 30, 20, step=5, key="num_strikes")
        pre_owned_n = st.slider("Pre-owned Units per Type (n)", 0, 10, 0, key="pre_owned")
        total_qty_per_deriv = st.slider("Total Quantity per Derivative", 10, 100, 10, step=5, key="total_qty")
        
        temp_deriv_hub = Derivative(S0, r, sigma, T, num_options, num_futures, num_strikes, total_qty_per_deriv)
        max_cash = min(100000, max(10000, int(temp_deriv_hub.get_market_cap() * 0.1)))
        initial_cash = st.slider("Initial Virtual Cash", 1000, max_cash, 10000, step=1000, key="initial_cash")
        mode = st.selectbox("Simulation Mode", ["Passive", "Neutral", "Aggro", "Very Aggro"], key="mode")

        col1, col2 = st.columns(2)
        with col1:
            start_sim = st.button("Start Simulation", key="start_sim")
        with col2:
            restart = st.button("Restart Simulation", key="restart_sim")

    # Initialize simulation
    if start_sim or (restart and 'initialized' in st.session_state):
        st.session_state.initialized = True
        st.session_state.price_path = simulate_gbm(S0, mu, sigma, T, T/steps, steps)
        st.session_state.current_step = 0
        st.session_state.deriv_hub = Derivative(S0, r, sigma, T, num_options, num_futures, num_strikes, total_qty_per_deriv)
        st.session_state.order_book = OrderBook(st.session_state.deriv_hub)
        st.session_state.user = UserTrader("User1", cash=initial_cash, deriv_hub=st.session_state.deriv_hub, pre_owned_n=pre_owned_n)
        st.session_state.bots = [VirtualTrader(f"Bot{i+1}", cash=initial_cash, deriv_hub=st.session_state.deriv_hub, all_traders=None, pre_owned_n=pre_owned_n) for i in range(num_virtual_traders)]
        st.session_state.lb1 = LiquidityBotCalls("LB1", cash=initial_cash, deriv_hub=st.session_state.deriv_hub)
        st.session_state.lb2 = LiquidityBotPuts("LB2", cash=initial_cash, deriv_hub=st.session_state.deriv_hub)
        st.session_state.lb3 = LiquidityBotFutures("LB3", cash=initial_cash, deriv_hub=st.session_state.deriv_hub)
        all_traders = [st.session_state.user] + st.session_state.bots
        for bot in st.session_state.bots:
            bot.all_traders = all_traders
        liquidity_bots = [st.session_state.lb1, st.session_state.lb2, st.session_state.lb3]
        all_derivs = st.session_state.deriv_hub.get_all_derivs()
        for lb in liquidity_bots:
            lb.assign_remaining(all_traders, all_derivs)
        st.session_state.running_auto = False
        st.session_state.auto_step = 0
        st.session_state.display_all_derivs = False
        st.session_state.trade_log = []
        st.session_state.init_log = []
        st.session_state.expiration_log = []
        st.session_state.sim_mode = mode
        st.session_state.pending_user_order = None
        st.session_state.executed_orders = set()
        st.success("Simulation started/restarted!")

    if not st.session_state.initialized:
        st.write("Please click 'Start Simulation' to begin.")
        return

    # Access session state
    deriv_hub = st.session_state.deriv_hub
    order_book = st.session_state.order_book
    user = st.session_state.user
    bots = st.session_state.bots
    lb1 = st.session_state.lb1
    lb2 = st.session_state.lb2
    lb3 = st.session_state.lb3
    sim_mode = st.session_state.sim_mode

    # Update bots if number changes
    if len(bots) != num_virtual_traders:
        all_traders = [user] + bots
        bots = [VirtualTrader(f"Bot{i+1}", cash=initial_cash, deriv_hub=deriv_hub, all_traders=all_traders, pre_owned_n=pre_owned_n) for i in range(num_virtual_traders)]
        st.session_state.bots = bots
        for bot in bots:
            bot.all_traders = all_traders
        liquidity_bots = [lb1, lb2, lb3]
        all_derivs = deriv_hub.get_all_derivs()
        for lb in liquidity_bots:
            lb.assign_remaining(all_traders, all_derivs)

    # Automation step
    def run_automation_step():
        if st.session_state.running_auto and st.session_state.current_step < len(st.session_state.price_path) - 1:
            st.session_state.current_step += 1
            st.session_state.auto_step += 1
            order_book.current_step = st.session_state.current_step
            current_price = st.session_state.price_path[st.session_state.current_step]
            deriv_hub.update_prices(current_price)
            user.expire_user_orders(st.session_state.current_step)
            for bot in bots:
                bot.simulate_trade(order_book, user, bots, sim_mode)
            lb1.force_trade(order_book, sim_mode)
            lb2.force_trade(order_book, sim_mode)
            lb3.force_trade(order_book, sim_mode)
            order_book.match_orders()
            time.sleep(0.5)
            st.rerun()

    # Automation controls
    col_auto, col_stop = st.columns(2)
    with col_auto:
        if st.button("Automate", key="auto_sim"):
            st.session_state.running_auto = True
            run_automation_step()
    with col_stop:
        if st.button("Stop Automation", key="stop_auto"):
            st.session_state.running_auto = False
            st.session_state.auto_step = 0

    if st.session_state.running_auto:
        run_automation_step()

    # Manual bot simulation
    if st.button("Simulate Bot Trades", key="simulate_bots"):
        st.session_state.running_auto = False
        st.session_state.auto_step = 0
        if st.session_state.current_step < len(st.session_state.price_path) - 1:
            st.session_state.current_step += 1
            order_book.current_step = st.session_state.current_step
            current_price = st.session_state.price_path[st.session_state.current_step]
            deriv_hub.update_prices(current_price)
            user.expire_user_orders(st.session_state.current_step)
            for bot in bots:
                bot.simulate_trade(order_book, user, bots, sim_mode)
            lb1.force_trade(order_book, sim_mode)
            lb2.force_trade(order_book, sim_mode)
            lb3.force_trade(order_book, sim_mode)
            order_book.match_orders()

    # Update current state
    current_price = st.session_state.price_path[st.session_state.current_step]
    deriv_hub.update_prices(current_price)
    order_book.current_step = st.session_state.current_step
    user.expire_user_orders(st.session_state.current_step)

    # Display trades
    trades_df = order_book.get_trades()
    if not trades_df.empty:
        with st.container():
            st.markdown(
                """
                <style>
                .trades-table {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    width: 300px;
                    max-height: 200px;
                    overflow-y: auto;
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    padding: 5px;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.markdown('<div class="trades-table">', unsafe_allow_html=True)
            st.subheader("Successful Trades")
            st.dataframe(
                trades_df[["Buyer", "Seller", "Derivative", "Quantity", "Price"]].style.format({"Price": "{:.2f}"}),
                height=150
            )
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
            .trades-table {
                position: absolute;
                top: 10px;
                right: 10px;
                width: 300px;
            }
            </style>
            <div class="trades-table">No trades yet.</div>
            """,
            unsafe_allow_html=True
        )

    # Display market state
    st.write(f"Current Underlying Price: ${current_price:.2f}")
    st.write(f"Simulation Step: {st.session_state.current_step}")
    st.write(f"Market Capitalization: ${deriv_hub.get_market_cap():.2f}")
    if st.session_state.trade_log:
        st.write("Trade Log (Last 5):", st.session_state.trade_log[-5:])
    if st.session_state.expiration_log:
        st.write("Expiration Log (Last 5):", st.session_state.expiration_log[-5:])
    if st.session_state.init_log:
        st.write("Initialization Log:", st.session_state.init_log)

    # Warn if simulation ends
    if st.session_state.current_step >= len(st.session_state.price_path) - 1:
        st.warning("Simulation Ended!")
        st.session_state.running_auto = False
        st.session_state.auto_step = 0

    # Show all derivatives toggle
    show_all_button = st.button("Show All Derivatives", key="show_all_derivs")
    if show_all_button:
        st.session_state.display_all_derivs = not st.session_state.display_all_derivs

    if st.session_state.display_all_derivs:
        st.subheader("All Derivatives")
        deriv_data = []
        all_traders = [user] + bots + [lb1, lb2, lb3]
        for deriv in deriv_hub.get_all_derivs():
            owners = [t.name for t in all_traders if deriv in t.positions and t.positions[deriv] > 0]
            total_qty = sum(t.positions.get(deriv, 0) for t in all_traders)
            price = deriv_hub.get_price(deriv)
            deriv_data.append({
                "Derivative": deriv,
                "Quantity Owned": total_qty,
                "Price per Share": f"{price:.2f}" if price is not None else "N/A",
                "Owner(s)": ", ".join(owners) if owners else "None"
            })
        st.dataframe(pd.DataFrame(deriv_data))

    # User and Bot columns
    user_col, bot_col = st.columns(2)

    with user_col:
        st.header("User Desk")
        all_derivs = deriv_hub.get_all_derivs()
        selected_user_deriv = st.selectbox("Select Derivative to Trade", all_derivs, key="user_deriv")
        user.selected_deriv = selected_user_deriv
        price = deriv_hub.get_price(selected_user_deriv)
        st.write(f"Selected Derivative Price: ${price:.2f}" if price is not None else "Price not yet calculated")
        quantity = st.number_input("Quantity to Trade", min_value=1, max_value=10, value=1, step=1, key="quantity")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Buy (User)", key="buy_user"):
                order_id = f"buy_{selected_user_deriv}_{quantity}_{st.session_state.current_step}"
                st.session_state.pending_user_order = {
                    "Side": "buy",
                    "Derivative": selected_user_deriv,
                    "Quantity": quantity,
                    "Price": price,
                    "OrderID": order_id,
                    "Status": "Staged"
                }
                st.session_state.trade_log.append(f"Staged Buy Order: {quantity} {selected_user_deriv} at ${price:.2f}")
        with col2:
            if st.button("Sell (User)", key="sell_user"):
                order_id = f"sell_{selected_user_deriv}_{quantity}_{st.session_state.current_step}"
                st.session_state.pending_user_order = {
                    "Side": "sell",
                    "Derivative": selected_user_deriv,
                    "Quantity": quantity,
                    "Price": price,
                    "OrderID": order_id,
                    "Status": "Staged"
                }
                st.session_state.trade_log.append(f"Staged Sell Order: {quantity} {selected_user_deriv} at ${price:.2f}")

        if st.button("Execute Trade", key="execute_trade"):
            if st.session_state.pending_user_order:
                order = st.session_state.pending_user_order
                order_id = order["OrderID"]
                if order_id not in st.session_state.executed_orders:
                    st.session_state.executed_orders.add(order_id)
                    st.session_state.current_step += 1
                    order_book.current_step = st.session_state.current_step
                    user.place_order(order_book, order["Derivative"], order["Price"], order["Quantity"], order["Side"])
                    order_book.match_orders()
                    matched = any(t["Buyer"] == "User1" or t["Seller"] == "User1" for t in order_book.get_trades().to_dict('records')[-1:])
                    if not matched:
                        st.session_state.trade_log.append(f"Order Failed: {order['Side']} {order['Quantity']} {order['Derivative']} at ${order['Price']:.2f}")
                    st.session_state.pending_user_order = None
                else:
                    st.session_state.trade_log.append(f"Order Already Executed: {order['Side']} {order['Quantity']} {order['Derivative']}")
            else:
                st.session_state.trade_log.append("No pending order to execute")

        st.write(f"User Virtual Cash: ${user.cash:.2f}")
        st.write(f"Number of Owned Derivatives: {len(user.positions)}")
        st.write("User Positions:", user.positions)

        st.subheader("Pending Order")
        if st.session_state.pending_user_order:
            pending_df = pd.DataFrame([st.session_state.pending_user_order])
            st.dataframe(pending_df[["Side", "Derivative", "Quantity", "Price", "Status"]].style.format({"Price": "{:.2f}"}))
        else:
            st.write("No pending order")

        st.subheader("User Trade Activity")
        if user.user_orders:
            st.write("Placed Orders:")
            for order in user.user_orders:
                if order["Status"] == "Pending":
                    order["Steps Remaining"] = max(0, 7 - (st.session_state.current_step - order["Step Placed"]))
            orders_df = pd.DataFrame(user.user_orders)
            st.dataframe(orders_df[["Side", "Derivative", "Quantity", "Price", "Status", "Steps Remaining"]].style.format({"Price": "{:.2f}"}))
        else:
            st.write("No orders placed yet.")

        user_trades = [t for t in order_book.get_trades().to_dict('records') if t["Buyer"] == "User1" or t["Seller"] == "User1"]
        if user_trades:
            st.write("Successful User Trades:")
            trades_df = pd.DataFrame(user_trades)
            st.dataframe(trades_df[["Buyer", "Seller", "Derivative", "Quantity", "Price"]].style.format({"Price": "{:.2f}"}))
        else:
            st.write("No successful trades yet.")

        st.subheader("Underlying Asset Price Path")
        underlying_df = pd.DataFrame({
            "Step": range(st.session_state.current_step + 1),
            "Price": st.session_state.price_path[:st.session_state.current_step + 1]
        })
        st.line_chart(underlying_df.set_index("Step"))

        st.subheader("Derivative Price History (Lazy Calc)")
        all_derivs = deriv_hub.get_all_derivs()
        selected_deriv_graph = st.selectbox("Select Derivative for Graph", all_derivs, key="deriv_graph")
        price_history = deriv_hub.get_lazy_price_history(selected_deriv_graph, st.session_state.price_path, st.session_state.current_step)
        if price_history and len(price_history) > 0:
            deriv_df = pd.DataFrame({
                "Step": range(len(price_history)),
                "Price": price_history
            })
            st.line_chart(deriv_df.set_index("Step"))
        else:
            st.write("No price history available yet.")

        st.subheader("Order Book")
        order_book_data = order_book.display()
        st.write("Bids (Buy Orders):")
        if not order_book_data["Bids"].empty:
            st.dataframe(order_book_data["Bids"].style.format({"Price": "{:.2f}"}))
        else:
            st.write("No bids yet.")
        st.write("Asks (Sell Orders):")
        if not order_book_data["Asks"].empty:
            st.dataframe(order_book_data["Asks"].style.format({"Price": "{:.2f}"}))
        else:
            st.write("No asks yet.")

    with bot_col:
        st.header("Bot Desk")
        for bot in bots:
            with st.expander(f"{bot.name} Details"):
                st.write(f"{bot.name} Cash: ${bot.cash:.2f}")
                st.write(f"Number of Owned Derivatives: {len(bot.positions)}")
                st.write(f"{bot.name} Positions:", bot.positions)

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
