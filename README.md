# Systematic-Trading-Strategies-on-Insider-Transactions

The Insider Trading.ipynb containts all the codes needed to perform the trading based on insider transcations information.

The strategy only consider the Acquired information or buy position from insider since it is obvious why the insider taking long position wherease there are several reasons why insider take short positions.

The dataset is extracted from Form 3, 4 and 5 on SEC EDGAR from 2006 to 2023. 

The .ipynb contains 2 rebalancing methods with 7 different trading strategies.
The rebalancing methods are train-test method and annual rebalancing method. The trading strategies are naive buy and hold, insider-company relationship, insider trade size, insider holding ratio, unique insider, co-location and winning insider strategies.

The results from backtester shown that annual rebalancing with consideration on holding ratio gives best trading performance of annual sharpe of 1.25
![image](https://github.com/Foktavianes/Systematic-Trading-Strategies-on-Insider-Transactions/assets/112449862/5ac89349-76ec-4485-ac4c-2a0a91c8207b)
