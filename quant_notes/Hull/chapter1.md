
* **Derivative**: financial instrument whose value depends on the values of other more basic underlying variables. Often, the underlying
variable is the price of an asset.

* Derivatives may be traded on **exchanges**, in which they need to have standardized, pre-defined conditions 
    - often, on exchanged traded derivatives, financial institutions act as **market makers** for common instruments,
    meaning they trade both the bid and ask and try to profit from spreads, volatility, etc.

* Derivatives are also often traded OTC, between financial institutions and corporate clients

### Forward Contracts

* **Forward contract** is arguably the simplest kind of derivative, which is
    an _agreement_ to buy or sell an asset at a certain future time, for a certain price.
    - one party of a forward contract is long, and agrees to buy, and the other is short, and agrees to sell (duh)
    - different from a spot contract, which is an agreement to buy/sell at the current market rate, "spot price"
    - The payoff for a long position is very simple, \\(S_T - K \\) where \\(S\\) is spot price and \\(K\\) is the delivery price
    - **Forward price** is the current market price for a particular forward contract, and this fluctuates over time
    - **Delivery price** is the agreed upon price of the asset specified by the contract, which is constant
    - The primary driving force for the difference between forward price and spot price is **interest rates**
    - Example, if one can borrow \\$300 at 5\% annualized and buy gold for \\$300, and short a contract of 1 yr for \\$340, this price inefficiency allows \\$25 of risk-free earnings on the trade
    - known as an **arbitrage**, and will likely get priced out of the market in due time 

* Forward contracts are always OTC, and do not have an associated premium, unlike, say, options contracts
    
### Futures Contracts
Essentially the same as forward contracts, but traded on an exchange with specific conditions of the contract,
e.g. 5000 bushels of corn with a specific price per bushel, with a September expiration.
The price will fluctuate as expected according to supply and demand. 
* The majority of futures contracts are closed out and do not lead to actual delivery.
* Delivery specification is done by the exchange, (really only important for commodities futures).
* Futures contracts are defined by their delivery month, and the exchange specifies a precise period for delivery.
* Futures contract prices converges to the spot price of the underlying asset as it approaches expiration

* Margin requirements: to prevent contract defaults
    - Entering into a position requires an amount of initial margin
    - The margin account is adjusted to reflect an investors gain/loss (known as daily settlement, or marking to market)
    - If an investors long futures contract trade decreases by X, their broker needs to pay the exchange clearing house X,
      and that money is passed onto the broker of an investor with a short position
    - A margin account never becomes negative (ideally) there is a maintenance margin, which is somewhat lower than the initial margin.
    - A drop below the maintenance margin level will trigger a margin call

The **volume** is the daily volume of the contract
The **open interest** is the total open contracts as of yesterdays close

### Hedging and Futures

* **Perfect hedge** completely eliminates risks, which is very rare.
* **Basis risk** in a hedging situation is the spot price of the asset to be hedged minus the futures price of the contract
* **hedge ratio** is the ratio of the size of the position taken in futures contract to the size of exposure
    - if the underlying of the futures is the same as the asset, the ratio is 1.0 
    - otherwise, picking a 1.0 ratio is not always optimal
    - the optimal choice is _minimizing_ the _variance_ of the hedged position
    - the minimum variance hedge ratio is the slope of the line which fits changes in spot \\(\Delta S\\) to changes in the futures \\(\Delta F\\) price. In other words, it represents the average change in spot for a particular change in the futures position
\\[h^* = \rho \frac{\sigma_S}{\sigma_F} \\]
    - Here, \\( \rho \\) is the correlation coefficient, and \\( \sigma_S \\) is the standard deviation of \\(\Delta S\\), likewise for \\( \sigma_F \\)
    - In practice, these parameters are estimated  by historical data on changes in spot and changes in the futures price. A set of time intervals are chosen, and values of \\(\Delta S\\) and \\(\Delta F\\) are observed





