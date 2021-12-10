### APRA SP530 compliant stress-testing model
This model has been developed to conduct a series of stress tests on multi-asset portfolios to comply with APRA standards. This model has been used by business as a self-service tool to produce various stress test reports for internal and external usage. User interface with message box/file browser is built inside the code for non-technical users.

##### Test 5.1. Monte-Carlo simulation based on asset class forecast
###### Input:
* asset class return
* asset class standard deviation
* asset class correlation matrix
* SAA. work with multiple portfolio allocations
* VaR level
* illiquid ratio. 0-100 represents how illiquid each asset class is to determine total portfolio illiquidity at VaR            
* number of simulations

###### Output:
* portfolio VaR
* asset class return contribution at VaR with portfolio illiquidity
* simulated asset class annual returns

##### Test 5.2. Monte-Carlo simulation based on distribution derived from historical data
###### Input:
* SAA.
* list of historical events. a list of date ranges selected by user
* illiquid ratio. 0-100 represents how illiquid each asset class is to determine total portfolio illiquidity at VaR            
* number of simulations
* asset class historical prices

###### Output:
* portfolio return on average based on simulation
* asset class return contribution with portfolio illiquidity
* simulated asset class annual returns

##### Test 5.3. Scenario test - historical events
###### Input:
* SAA.
* list of historical events. a list of date ranges selected by user          
* asset class historical prices

###### Output:
* portfolio return at maximal drawdown
* date range when portfolios hit maximal drawdown
* asset class allocation at maximal drawdown

##### Test 7. test portfolio performance at VaR together with various redemption relationship
Basically test 5.1 plus redemption to test when portfolio underperforms which may trigger high redemption request.


#### Assumptions:
* Monte Carlo simulation assumes returns are normally distributed
* Correlation matrix only measures linear dependency and 1-1 relationship (instead of multi-collinearity)
* Result reliability heavily depends on forward-looking values provided by users
