# DAO-Treasury-Risk-Runway-Stress-Tester

=== Monte Carlo Risk Report (1-month) ===

initial_value       : $9,000,000
horizon_months      : 1.00
VaR_95              : $1,156,514
CVaR_95             : $1,393,228
worst_value         : $6,611,087
p01_value           : $7,453,991
p05_value           : $7,843,486
median_value        : $9,025,737
worst_runway_months : 26.44
p05_runway_months   : 31.37

=== Deterministic Scenarios ===

Scenario: ETH -50%, BTC -40%, DAO token -70% (risk-off shock)
  Initial Value     : $9,000,000
  Stressed Value    : $6,876,900
  Loss              : $2,123,100
  Drawdown          : 23.59%
  Runway (months)   : 27.51
  Risk Label        : GREEN

Scenario: Stablecoin depeg: USDC 0.92, DAI 0.95 (temporary impairment)
  Initial Value     : $9,000,000
  Stressed Value    : $8,621,630
  Loss              : $378,370
  Drawdown          : 4.20%
  Runway (months)   : 34.49
  Risk Label        : GREEN

Scenario: Protocol incident: LP_Aave loses 30% principal + protocol haircut
  Initial Value     : $9,000,000
  Stressed Value    : $8,514,100
  Loss              : $485,900
  Drawdown          : 5.40%
  Runway (months)   : 34.06
  Risk Label        : GREEN

Scenario: Liquidity crisis: forced unwind haircuts (no price move)
  Initial Value     : $9,000,000
  Stressed Value    : $8,911,000
  Loss              : $89,000
  Drawdown          : 0.99%
  Runway (months)   : 35.64
  Risk Label        : GREEN

=== Warnings / Triggers ===
No triggers breached (runway stayed above 18 months in tested outputs).

=== Plain-English Risk Summary ===

Overall Treasury Risk (1-Month Outlook):
- The treasury starts at $9,000,000 with monthly operating costs of $250,000.
- In 95% of realistic short-term market conditions, losses should not exceed about $1,156,514.
- In unusually bad market conditions (the tail), the average loss is closer to $1,393,228.
- In the worst simulated month, the treasury could fall to $6,611,087, which is about 26.4 months of runway.
- Bottom line: the organization remains financially viable even under extreme short-term stress.

Specific Stress Scenarios:

Scenario: ETH -50%, BTC -40%, DAO token -70% (risk-off shock)
- Risk level (runway-based): GREEN
- Estimated loss: $2,123,100 (23.6% drawdown).
- Remaining treasury: $6,876,900 = about 27.5 months of operating runway.
- Interpretation: Uncomfortable but manageable; no emergency action required.

Scenario: Stablecoin depeg: USDC 0.92, DAI 0.95 (temporary impairment)
- Risk level (runway-based): GREEN
- Estimated loss: $378,370 (4.2% drawdown).
- Remaining treasury: $8,621,630 = about 34.5 months of operating runway.
- Interpretation: Uncomfortable but manageable; no emergency action required.

Scenario: Protocol incident: LP_Aave loses 30% principal + protocol haircut
- Risk level (runway-based): GREEN
- Estimated loss: $485,900 (5.4% drawdown).
- Remaining treasury: $8,514,100 = about 34.1 months of operating runway.
- Interpretation: Uncomfortable but manageable; no emergency action required.

Scenario: Liquidity crisis: forced unwind haircuts (no price move)
- Risk level (runway-based): GREEN
- Estimated loss: $89,000 (1.0% drawdown).
- Remaining treasury: $8,911,000 = about 35.6 months of operating runway.
- Interpretation: Uncomfortable but manageable; no emergency action required.

Overall Conclusion:
- The primary risk driver is broad market drawdowns across correlated crypto assets.
- Stablecoin or single-protocol events matter, but are less existential when exposure is sized correctly.
- The key governance question is: how much runway do we want to guarantee under stress?
