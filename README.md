Overview:  

Note: The Streamlit app may take 30–60 seconds to wake up on first load due to free hosting cold-start. 
Short demo showing live order matching, market maker behavior, and state updates in the simulator.

This project implements a virtual derivatives market simulator designed to model option pricing, market making behavior, order matching, and trader interaction in a controlled synthetic environment. The simulator focuses on European call and put options priced using the Black–Scholes framework and evolves toward a multi-agent system with virtual traders, market makers, an order book, and execution logic.
The objective of the project is not predictive accuracy, but to understand and simulate the mechanics of derivatives markets, including pricing, liquidity provision, order flow, and system-level constraints.

Problem Statement: 

Real-world derivatives markets involve complex interactions between pricing models, liquidity providers, and traders under various constraints. This project aims to simulate these interactions by building a simplified but extensible virtual market where options can be priced, listed, traded, and settled while exposing practical design and implementation challenges.

Core Components:


Pricing Engine: 

The simulator uses a Black–Scholes-based pricing model for European options. The pricing logic evolves over multiple iterations to handle parameter edge cases, option types, and stability issues.

Market Makers:

Multiple market maker agents are implemented with distinct behaviors and inventory logic. These agents provide liquidity, quote prices, and interact with incoming orders.

Order Book and Matching Engine:

An order book tracks outstanding buy and sell orders. A matching function executes trades based on price-time priority and available liquidity.

Virtual Traders:

Trader agents interact with the market by placing buy and sell orders using virtual capital, enabling realistic trading flows.

System Interfaces:

A terminal-style interface and logs provide visibility into trades, pricing, and system state during simulation runs.

Approach:


Initial Design:

The simulator began with a simple Black–Scholes pricing function and a single market maker holding all derivatives. Trades were conceptual rather than executable, and the system lacked capital constraints and order matching.

Iterative Expansion:

As complexity increased, the system was refactored into modular components including pricing, market making, order management, and execution. A lazy calculation approach was adopted to manage computational and architectural complexity.

Challenges Faced and Design Decisions:


Numerical Instability in the Initial Black–Scholes Model:

The initial implementation of the Black–Scholes model did not include proper error handling. Certain parameter combinations, such as zero or near-zero volatility, invalid maturities, or extreme strikes, caused runtime errors and numerical instability.

Resolution:

Try-except blocks and parameter validation checks were introduced to ensure robustness. Invalid parameter combinations were handled gracefully rather than crashing the simulation, improving system stability.

Missing Option Type Handling:

The first version of the pricing engine did not explicitly distinguish between call and put options, limiting the realism and flexibility of the simulator.

Resolution:

An explicit option type parameter was added to the Black–Scholes model, enabling correct pricing logic for both calls and puts and allowing the simulator to support a broader derivative universe.

Flawed Initial Market Maker Design:

The initial market maker design assumed that the market maker possessed all derivatives and that users had no virtual capital. As a result, meaningful trades could not occur because users were unable to buy or sell instruments, and inventory dynamics were unrealistic.

Resolution:

Virtual capital was introduced for users, and market maker inventory logic was revised. This enabled actual trade execution, inventory transfer, and more realistic liquidity provision behavior.

Limited Derivative Universe:

Early versions of the simulator included only 20 strike prices and a total of 50 call and put derivatives. This limited market depth and reduced the realism of price discovery and trading behavior.

Resolution:

The design was generalized to support scalable derivative generation. Although complexity constraints later required a controlled universe, the system architecture was adapted to allow extensibility.

System Complexity and the Shift to Lazy Evaluation:

As features such as traders, order books, logs, and multiple market makers were added, eager computation across all components led to increasing complexity and difficult-to-debug interactions.

Design Decision:

A lazy calculation approach was adopted, where computations were performed only when required rather than eagerly across the entire system. This significantly improved manageability and reduced unintended side effects across components.

Order Book, Log Book, and Interface Synchronization Issues:

Introducing multiple subsystems such as the order book, log book, terminal interface, and virtual traders led to synchronization issues. Problems included missing outputs, incorrect logs, and mismatches between executed trades and displayed system state.

Resolution:

Each subsystem was debugged and validated independently before reintegration. Output consistency checks were added, and logging logic was refined to ensure alignment between internal state and visible outputs.

Order Matching Logic Limitations:

Initial trade execution logic lacked a robust matching mechanism, resulting in incomplete or incorrect trade execution under certain conditions.

Resolution:

A dedicated match_order function was implemented to handle price matching, execution priority, and trade settlement. This significantly improved the correctness and realism of trade execution.

Evolution of Market Makers:

The initial single market maker was insufficient to model competitive liquidity provision. Additional market makers (lb1, lb2, lb3) were introduced with progressively improved logic, enabling better price competition, inventory distribution, and execution dynamics.

Outcome:

This multi-market-maker setup allowed the simulator to better reflect real-world liquidity fragmentation and competition.

Results:

The final system supports a functioning virtual derivatives market with option pricing, multiple market makers, order matching, virtual traders, and detailed logging. The simulator successfully demonstrates how pricing models, liquidity provision, and execution logic interact under practical constraints.

Key Learnings:

The project highlighted the importance of robust error handling in financial models, the necessity of capital and inventory constraints for realistic trading, and the complexity introduced by multi-component system design. It also reinforced the value of iterative development and architectural flexibility in simulation-heavy systems.

Future Improvements
Potential extensions include stochastic underlying price simulation, volatility surface modeling, more advanced market maker strategies, latency modeling, and risk metrics such as Greeks and PnL tracking.

