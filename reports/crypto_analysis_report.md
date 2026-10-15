# Cryptocurrency Market Volatility & Trend Analysis

> This is a data-science / statistical analysis of historical market data. Trend labels, volatility rankings and outlier flags are descriptive statistics, **not financial advice or trading recommendations**. Cryptocurrency prices are highly volatile and past behaviour does not predict future results.

| | |
|---|---|
| Generated | 2026-09-16 14:52 UTC |
| Data source | CoinGecko API (`/coins/markets`, `/coins/{id}/market_chart`) |
| Window | 2026-09-11 07:00 → 2026-09-16 09:53 UTC (latest 5 days) |
| Resolution | hourly observations (median interval 60 min, original API granularity) |
| Coins analysed | 100 (history retrieved for 100 of 100 requested) |
| Observations | 12,064 price points |

## 1. Key findings

- **Largest market cap:** Bitcoin (BTC) at $1.52T.
- **Most volatile:** Bitway (BTW) with σ = 4.866% per period (5-day return +51.28%).
- **Least volatile:** BlackRock USD Institutional Digital Liquidity Fund (BUIDL) with σ = 0.0000% per period.
- **Highest 5-day return:** Akedo (AKE) at +100.83%.
- **Lowest 5-day return:** Polkadot (DOT) at -17.72%.
- **Breadth:** 24 of 100 coins rose over the window; the median 5-day return was -2.04%.
- **Trend labels at the latest observation:** Bullish 20 · Bearish 24 · Neutral 56.
- **Outliers (|Z| > 2):** 611 of 11,844 returns (5.16%; a normal distribution implies ≈4.55%) — 304 positive, 307 negative.
- **Extreme outliers (|Z| > 3):** 199 (1.68% vs ≈0.27% under normality), consistent with fat-tailed crypto returns.
- **Most outliers:** OKB (OKB) with 10 (2 extreme).
- **Unusually high volatility (σ above Q3 + 1.5×IQR = 1.837%):** Bitway (BTW) 4.866%, Akedo (AKE) 3.515%.
- **Isolation Forest (multivariate) unusual coins:** Bitway (BTW), Akedo (AKE), Ether.fi (ETHFI), Rain (RAIN), JUST (JST).
- **Near-constant assets:** 23 of 100 coins (stablecoins and tokenized money-market/treasury funds) moved less than 0.5% across the entire window. They account for 109 of the 611 flagged outliers, because for an asset whose σ is ~0.003% a 0.03% blip is a 10σ event. Ranking purely by |Z| is dominated by these statistically-extreme-but-economically-trivial moves, so the largest *material* outliers are reported separately.

## 2. Top 10 most volatile cryptocurrencies

|   Volatility Rank | Name                    | Symbol   | Std Dev (%)   | Average Return (%)   | 5-Day Return (%)   | Current Price (USD)   |   Outlier Count |
|------------------:|:------------------------|:---------|:--------------|:---------------------|:-------------------|:----------------------|----------------:|
|                 1 | Bitway                  | BTW      | 4.866%        | +0.465%              | +51.283%           | $0.7685               |               7 |
|                 2 | Akedo                   | AKE      | 3.515%        | +0.649%              | +100.831%          | $0.02852              |               8 |
|                 3 | Ether.fi                | ETHFI    | 1.595%        | -0.106%              | -13.343%           | $0.5858               |               9 |
|                 4 | Arbitrum                | ARB      | 1.563%        | +0.046%              | +4.220%            | $0.1509               |               6 |
|                 5 | Venice Token            | VVV      | 1.534%        | -0.066%              | -8.942%            | $21.89                |               4 |
|                 6 | Filecoin                | FIL      | 1.413%        | +0.026%              | +1.991%            | $0.8077               |               7 |
|                 7 | Lighter                 | LIT      | 1.362%        | -0.038%              | -5.546%            | $4.26                 |               9 |
|                 8 | Pump.fun                | PUMP     | 1.320%        | -0.047%              | -6.506%            | $0.00353              |               6 |
|                 9 | World Liberty Financial | WLFI     | 1.288%        | +0.028%              | +2.370%            | $0.05701              |               5 |
|                10 | Zcash                   | ZEC      | 1.223%        | +0.071%              | +7.894%            | $1,191                |               5 |

## 3. Top 10 least volatile cryptocurrencies

|   Volatility Rank | Name                                                 | Symbol   | Std Dev (%)   | Average Return (%)   | 5-Day Return (%)   | Current Price (USD)   |   Outlier Count |
|------------------:|:-----------------------------------------------------|:---------|:--------------|:---------------------|:-------------------|:----------------------|----------------:|
|               100 | BlackRock USD Institutional Digital Liquidity Fund   | BUIDL    | 0.000%        | +0.000%              | +0.000%            | $1.00                 |               0 |
|                99 | Janus Henderson Anemoy Treasury Fund                 | JTRSY    | 0.002%        | +0.000%              | +0.030%            | $1.12                 |               2 |
|                98 | Circle USYC                                          | USYC     | 0.003%        | +0.000%              | +0.043%            | $1.14                 |               3 |
|                97 | Invesco Short Duration US Government Securities Fund | USTB     | 0.003%        | +0.000%              | +0.048%            | $11.21                |               3 |
|                96 | USDC                                                 | USDC     | 0.003%        | -0.000%              | -0.011%            | $0.9997               |               6 |
|                95 | Janus Henderson Anemoy AAA CLO Fund                  | JAAA     | 0.004%        | +0.001%              | +0.064%            | $1.05                 |               2 |
|                94 | Blockchain Capital                                   | BCAP     | 0.004%        | +0.000%              | +0.047%            | $106.54               |               1 |
|                93 | Tether                                               | USDT     | 0.004%        | -0.000%              | -0.028%            | $0.9993               |               8 |
|                92 | Ethena USDe                                          | USDE     | 0.006%        | -0.000%              | -0.035%            | $0.9991               |               6 |
|                91 | Ripple USD                                           | RLUSD    | 0.006%        | +0.000%              | +0.027%            | $1.00                 |               6 |

## 4. Outlier report

611 observations were flagged in total (full list: `outlier_report.csv`).
109 of them belong to near-constant assets - stablecoins and tokenized money-market funds whose
sigma is so small that a few hundredths of a percent registers as a 10-sigma event. Those are real
statistical outliers but not market events, so the table below ranks the **502 material
outliers** (near-constant assets excluded) by |Z|. Examples from this run:

```text
World Liberty Financial | 2026-09-11 11:00 | -9.42% | Z=-7.34 | Negative Outlier
Ethereum | 2026-09-11 14:00 | +4.68% | Z=7.04 | Positive Outlier
Rain | 2026-09-15 17:00 | +7.57% | Z=6.89 | Positive Outlier
KuCoin | 2026-09-11 13:00 | +2.79% | Z=6.20 | Positive Outlier
LEO Token | 2026-09-15 15:00 | -1.94% | Z=-6.10 | Negative Outlier
```

| Cryptocurrency          | Symbol   | Date/Time (UTC)   | Price (USD)   | Return (%)   |   Z-score | Outlier Type     | Severity   |
|:------------------------|:---------|:------------------|:--------------|:-------------|----------:|:-----------------|:-----------|
| World Liberty Financial | WLFI     | 2026-09-11 11:00  | $0.04962      | -9.424%      |     -7.34 | Negative Outlier | Extreme    |
| Ethereum                | ETH      | 2026-09-11 14:00  | $2,627        | +4.681%      |      7.04 | Positive Outlier | Extreme    |
| Rain                    | RAIN     | 2026-09-15 17:00  | $0.0136       | +7.568%      |      6.89 | Positive Outlier | Extreme    |
| KuCoin                  | KCS      | 2026-09-11 13:00  | $7.05         | +2.794%      |      6.2  | Positive Outlier | Extreme    |
| LEO Token               | LEO      | 2026-09-15 15:00  | $8.77         | -1.941%      |     -6.1  | Negative Outlier | Extreme    |
| XRP                     | XRP      | 2026-09-15 19:00  | $1.33         | -5.645%      |     -5.98 | Negative Outlier | Extreme    |
| Stellar                 | XLM      | 2026-09-15 19:00  | $0.1818       | -5.612%      |     -5.87 | Negative Outlier | Extreme    |
| WhiteBIT Coin           | WBT      | 2026-09-11 14:00  | $82.73        | +2.396%      |      5.62 | Positive Outlier | Extreme    |
| Tether Gold             | XAUT     | 2026-09-11 13:00  | $4,386        | +1.252%      |      5.56 | Positive Outlier | Extreme    |
| Arbitrum                | ARB      | 2026-09-15 16:00  | $0.1495       | +8.713%      |      5.55 | Positive Outlier | Extreme    |
| PAX Gold                | PAXG     | 2026-09-11 13:00  | $4,389        | +1.216%      |      5.46 | Positive Outlier | Extreme    |
| Pi Network              | PI       | 2026-09-16 02:00  | $0.08618      | -3.921%      |     -5.46 | Negative Outlier | Extreme    |
| Sky                     | SKY      | 2026-09-11 17:00  | $0.06262      | +4.579%      |      5.3  | Positive Outlier | Extreme    |
| Ethena                  | ENA      | 2026-09-11 14:00  | $0.1557       | +5.394%      |      5.25 | Positive Outlier | Extreme    |
| Pepe                    | PEPE     | 2026-09-11 14:00  | $0.000003518  | +4.710%      |      5.19 | Positive Outlier | Extreme    |
| Solana                  | SOL      | 2026-09-11 14:00  | $104.78       | +3.139%      |      5.14 | Positive Outlier | Extreme    |
| Ethereum Classic        | ETC      | 2026-09-11 14:00  | $8.00         | +3.897%      |      5.09 | Positive Outlier | Extreme    |
| Shiba Inu               | SHIB     | 2026-09-11 14:00  | $0.000005342  | +3.635%      |      5.05 | Positive Outlier | Extreme    |
| ​​Stable                  | STABLE   | 2026-09-11 12:00  | $0.02733      | -4.857%      |     -5.05 | Negative Outlier | Extreme    |
| Bitway                  | BTW      | 2026-09-14 11:00  | $0.5899       | -24.043%     |     -5.04 | Negative Outlier | Extreme    |
| ​​Stable                  | STABLE   | 2026-09-11 14:00  | $0.02907      | +4.659%      |      5    | Positive Outlier | Extreme    |
| Pump.fun                | PUMP     | 2026-09-12 18:00  | $0.003911     | +6.518%      |      4.98 | Positive Outlier | Extreme    |
| Hyperliquid             | HYPE     | 2026-09-11 13:00  | $81.93        | +3.436%      |      4.92 | Positive Outlier | Extreme    |
| BNB                     | BNB      | 2026-09-11 14:00  | $736.13       | +1.780%      |      4.87 | Positive Outlier | Extreme    |
| Filecoin                | FIL      | 2026-09-13 15:00  | $0.9135       | +6.905%      |      4.87 | Positive Outlier | Extreme    |

### Largest |Z| overall, including near-constant assets

Shown to make the distinction concrete: the biggest Z-scores in the data set belong to assets that barely
moved in price terms.

| Cryptocurrency                                       | Symbol   | Date/Time (UTC)   | Price (USD)   | Return (%)   |   Z-score | Outlier Type     | Severity   | Low-Variation Asset   |
|:-----------------------------------------------------|:---------|:------------------|:--------------|:-------------|----------:|:-----------------|:-----------|:----------------------|
| Blockchain Capital                                   | BCAP     | 2026-09-15 03:00  | $106.54       | +0.047%      |     10.91 | Positive Outlier | Extreme    | True                  |
| Circle USYC                                          | USYC     | 2026-09-11 13:00  | $1.14         | +0.028%      |     10.15 | Positive Outlier | Extreme    | True                  |
| Janus Henderson Anemoy AAA CLO Fund                  | JAAA     | 2026-09-14 09:00  | $1.05         | +0.041%      |     10.06 | Positive Outlier | Extreme    | True                  |
| Janus Henderson Anemoy Treasury Fund                 | JTRSY    | 2026-09-14 09:00  | $1.12         | +0.019%      |      9.98 | Positive Outlier | Extreme    | True                  |
| Invesco Short Duration US Government Securities Fund | USTB     | 2026-09-15 14:00  | $11.21        | +0.029%      |      9.84 | Positive Outlier | Extreme    | True                  |
| World Liberty Financial                              | WLFI     | 2026-09-11 11:00  | $0.04962      | -9.424%      |     -7.34 | Negative Outlier | Extreme    | False                 |
| Ethereum                                             | ETH      | 2026-09-11 14:00  | $2,627        | +4.681%      |      7.04 | Positive Outlier | Extreme    | False                 |
| Rain                                                 | RAIN     | 2026-09-15 17:00  | $0.0136       | +7.568%      |      6.89 | Positive Outlier | Extreme    | False                 |
| Usual USD                                            | USD0     | 2026-09-16 06:59  | $0.9991       | +0.130%      |      6.61 | Positive Outlier | Extreme    | True                  |
| Falcon USD                                           | USDF     | 2026-09-12 15:00  | $0.9977       | +0.110%      |      6.59 | Positive Outlier | Extreme    | True                  |

## 5. Risk vs return quadrants

Split at the cross-sectional medians of standard deviation and mean return.

| Quadrant                      |   Coins |
|:------------------------------|--------:|
| High return / Low volatility  |      36 |
| Low return / High volatility  |      36 |
| Low return / Low volatility   |      14 |
| High return / High volatility |      14 |

## 6. Statistical summary (market-cap order)

Sortable copy: `crypto_analysis_report.csv` (or `python main.py --sort-by volatility|return|outliers`).

|   Rank | Cryptocurrency                                       | Symbol     | Price (USD)   | 5-Day Return (%)   | Mean Return (%)   | Std Dev (%)   | MA(3)        | MA(5)        | Trend   |   Outlier Count |
|-------:|:-----------------------------------------------------|:-----------|:--------------|:-------------------|:------------------|:--------------|:-------------|:-------------|:--------|----------------:|
|      1 | Bitcoin                                              | BTC        | $75,801       | -1.584%            | -0.013%           | 0.372%        | $75,725      | $75,811      | Neutral |               7 |
|      2 | Ethereum                                             | ETH        | $2,401        | -2.737%            | -0.021%           | 0.668%        | $2,399       | $2,401       | Neutral |               5 |
|      3 | Tether                                               | USDT       | $0.9993       | -0.028%            | -0.000%           | 0.004%        | $0.9993      | $0.9994      | Neutral |               8 |
|      4 | BNB                                                  | BNB        | $709.63       | -0.435%            | -0.003%           | 0.366%        | $708.45      | $710.75      | Neutral |               3 |
|      5 | XRP                                                  | XRP        | $1.29         | -3.929%            | -0.029%           | 0.939%        | $1.29        | $1.29        | Neutral |               6 |
|      6 | USDC                                                 | USDC       | $0.9997       | -0.011%            | -0.000%           | 0.003%        | $0.9997      | $0.9997      | Neutral |               6 |
|      7 | Solana                                               | SOL        | $97.22        | -2.187%            | -0.017%           | 0.614%        | $97.02       | $97.10       | Neutral |               7 |
|      8 | TRON                                                 | TRX        | $0.3349       | -1.077%            | -0.009%           | 0.220%        | $0.3347      | $0.3347      | Neutral |               6 |
|      9 | Figure Heloc                                         | FIGR_HELOC | $1.00         | -3.052%            | -0.033%           | 0.785%        | $1.00        | $1.00        | Neutral |               6 |
|     10 | Zcash                                                | ZEC        | $1,191        | +7.894%            | +0.071%           | 1.223%        | $1,186       | $1,181       | Bullish |               5 |
|     11 | Hyperliquid                                          | HYPE       | $77.52        | -2.488%            | -0.019%           | 0.702%        | $77.55       | $77.48       | Neutral |               5 |
|     12 | Dogecoin                                             | DOGE       | $0.07979      | -4.636%            | -0.038%           | 0.589%        | $0.07963     | $0.07982     | Neutral |               7 |
|     13 | USDS                                                 | USDS       | $0.9993       | -0.054%            | -0.000%           | 0.006%        | $0.9993      | $0.9994      | Bearish |               7 |
|     14 | Rain                                                 | RAIN       | $0.01375      | -12.381%           | -0.104%           | 1.114%        | $0.01378     | $0.01387     | Bearish |               6 |
|     15 | Monero                                               | XMR        | $502.47       | -1.641%            | -0.012%           | 0.636%        | $505.22      | $505.71      | Bearish |               8 |
|     16 | WhiteBIT Coin                                        | WBT        | $77.97        | -2.293%            | -0.018%           | 0.429%        | $77.87       | $77.92       | Neutral |               5 |
|     17 | LEO Token                                            | LEO        | $8.88         | -2.393%            | -0.020%           | 0.315%        | $8.89        | $8.89        | Neutral |               6 |
|     18 | Chainlink                                            | LINK       | $10.82        | -5.340%            | -0.043%           | 0.685%        | $10.78       | $10.80       | Neutral |               6 |
|     19 | Cardano                                              | ADA        | $0.1943       | -4.670%            | -0.037%           | 0.777%        | $0.1938      | $0.1946      | Neutral |               6 |
|     20 | Stellar                                              | XLM        | $0.1757       | +0.328%            | +0.007%           | 0.958%        | $0.1751      | $0.1756      | Neutral |               5 |
|     21 | Ethena USDe                                          | USDE       | $0.9991       | -0.035%            | -0.000%           | 0.006%        | $0.9991      | $0.9991      | Bullish |               6 |
|     22 | Dai                                                  | DAI        | $0.9999       | -0.003%            | -0.000%           | 0.012%        | $1           | $1.00        | Bearish |               4 |
|     23 | Bitcoin Cash                                         | BCH        | $219.49       | -3.225%            | -0.025%           | 0.716%        | $219.76      | $219.54      | Neutral |               6 |
|     24 | USD1                                                 | USD1       | $0.9992       | -0.030%            | -0.000%           | 0.008%        | $0.9992      | $0.9992      | Neutral |               6 |
|     25 | Litecoin                                             | LTC        | $50.99        | -3.703%            | -0.030%           | 0.567%        | $51.08       | $51.08       | Neutral |               6 |
|     26 | Uniswap                                              | UNI        | $6.28         | +2.113%            | +0.025%           | 1.216%        | $6.35        | $6.33        | Neutral |               7 |
|     27 | Gram (prev. Toncoin)                                 | GRAM       | $1.32         | -2.833%            | -0.023%           | 0.416%        | $1.31        | $1.31        | Bullish |               6 |
|     28 | Canton                                               | CC         | $0.09089      | -8.125%            | -0.068%           | 0.711%        | $0.0915      | $0.09142     | Neutral |               6 |
|     29 | Global Dollar                                        | USDG       | $1.00         | +0.024%            | +0.000%           | 0.006%        | $1.00        | $1.00        | Bullish |               7 |
|     30 | Avalanche                                            | AVAX       | $7.30         | -2.948%            | -0.023%           | 0.691%        | $7.32        | $7.30        | Neutral |               8 |
|     31 | NEAR Protocol                                        | NEAR       | $2.34         | -5.711%            | -0.043%           | 1.140%        | $2.34        | $2.34        | Neutral |               4 |
|     32 | Hedera                                               | HBAR       | $0.07442      | -0.811%            | -0.004%           | 0.716%        | $0.07447     | $0.07441     | Neutral |               7 |
|     33 | Shiba Inu                                            | SHIB       | $0.000004895  | -3.726%            | -0.029%           | 0.725%        | $0.000004911 | $0.000004916 | Bearish |              10 |
|     34 | Sui                                                  | SUI        | $0.6875       | -6.704%            | -0.055%           | 0.762%        | $0.6902      | $0.6897      | Neutral |               5 |
|     35 | PayPal USD                                           | PYUSD      | $0.9995       | -0.024%            | -0.000%           | 0.006%        | $0.9996      | $0.9996      | Neutral |               4 |
|     36 | Tether Gold                                          | XAUT       | $4,322        | -0.686%            | -0.005%           | 0.226%        | $4,322       | $4,321       | Neutral |               5 |
|     37 | BlackRock USD Institutional Digital Liquidity Fund   | BUIDL      | $1.00         | +0.000%            | +0.000%           | 0.000%        | $1.00        | $1.00        | Neutral |               0 |
|     38 | Cronos                                               | CRO        | $0.05535      | -1.890%            | -0.014%           | 0.691%        | $0.05546     | $0.05552     | Bearish |               5 |
|     39 | Circle USYC                                          | USYC       | $1.14         | +0.043%            | +0.000%           | 0.003%        | $1.14        | $1.14        | Neutral |               3 |
|     40 | MemeCore                                             | M          | $1.10         | -5.993%            | -0.048%           | 0.798%        | $1.11        | $1.12        | Bearish |               6 |
|     41 | Bittensor                                            | TAO        | $216.86       | -8.123%            | -0.068%           | 0.744%        | $217.50      | $217.82      | Bearish |               7 |
|     42 | Ripple USD                                           | RLUSD      | $1.00         | +0.027%            | +0.000%           | 0.006%        | $1.00        | $1.00        | Bullish |               6 |
|     43 | OKB                                                  | OKB        | $110.96       | -0.212%            | -0.000%           | 0.510%        | $111.06      | $111.15      | Bearish |              10 |
|     44 | Ondo US Dollar Yield                                 | USDY       | $1.15         | +0.061%            | +0.001%           | 0.131%        | $1.15        | $1.15        | Neutral |               7 |
|     45 | Bitway                                               | BTW        | $0.7685       | +51.283%           | +0.465%           | 4.866%        | $0.7674      | $0.7382      | Bullish |               7 |
|     46 | PAX Gold                                             | PAXG       | $4,327        | -0.655%            | -0.005%           | 0.224%        | $4,327       | $4,327       | Bullish |               5 |
|     47 | Aster                                                | ASTER      | $0.6776       | -4.086%            | -0.032%           | 0.688%        | $0.6782      | $0.6775      | Neutral |               7 |
|     48 | World Liberty Financial                              | WLFI       | $0.05701      | +2.370%            | +0.028%           | 1.288%        | $0.05704     | $0.05712     | Bearish |               5 |
|     49 | Mantle                                               | MNT        | $0.5435       | -5.950%            | -0.049%           | 0.650%        | $0.5445      | $0.5448      | Bearish |              10 |
|     50 | Aave                                                 | AAVE       | $120.43       | -2.244%            | -0.016%           | 0.759%        | $120.85      | $120.85      | Neutral |               6 |
|     51 | Pump.fun                                             | PUMP       | $0.00353      | -6.506%            | -0.047%           | 1.320%        | $0.003513    | $0.00349     | Bullish |               6 |
|     52 | Polkadot                                             | DOT        | $0.9483       | -17.718%           | -0.159%           | 0.775%        | $0.951       | $0.9507      | Neutral |               8 |
|     53 | Ondo                                                 | ONDO       | $0.3281       | -6.267%            | -0.051%           | 0.757%        | $0.3284      | $0.3282      | Neutral |               7 |
|     54 | USDD                                                 | USDD       | $0.9988       | +0.043%            | +0.000%           | 0.038%        | $0.9988      | $0.9988      | Neutral |               9 |
|     55 | HTX DAO                                              | HTX        | $0.000001661  | -1.642%            | -0.013%           | 0.283%        | $0.000001658 | $0.000001658 | Bullish |               6 |
|     56 | Ethena                                               | ENA        | $0.1398       | -4.234%            | -0.031%           | 1.032%        | $0.1402      | $0.1402      | Neutral |               4 |
|     57 | Morpho                                               | MORPHO     | $2.07         | -10.805%           | -0.092%           | 0.799%        | $2.07        | $2.07        | Bearish |               7 |
|     58 | Pepe                                                 | PEPE       | $0.000003374  | +1.832%            | +0.019%           | 0.905%        | $0.000003383 | $0.000003382 | Neutral |               7 |
|     59 | United Stables                                       | U          | $0.9994       | +0.020%            | +0.000%           | 0.010%        | $0.9995      | $0.9995      | Neutral |               7 |
|     60 | Spiko Amundi Overnight Swap Fund (EUR)               | EURSAFO    | $1.17         | -0.511%            | -0.004%           | 0.044%        | $1.17        | $1.17        | Bullish |               9 |
|     61 | USDGO                                                | USDGO      | $1.00         | -0.008%            | -0.000%           | 0.011%        | $1.00        | $1.00        | Bearish |               4 |
|     62 | Internet Computer                                    | ICP        | $2.53         | -8.180%            | -0.068%           | 0.817%        | $2.52        | $2.50        | Bullish |               8 |
|     63 | Falcon USD                                           | USDF       | $0.9958       | -0.048%            | -0.000%           | 0.017%        | $0.9958      | $0.9959      | Bearish |               3 |
|     64 | Sky                                                  | SKY        | $0.05995      | +0.313%            | +0.006%           | 0.864%        | $0.06042     | $0.06055     | Bearish |               3 |
|     65 | Bitget Token                                         | BGB        | $1.89         | -2.965%            | -0.024%           | 0.371%        | $1.89        | $1.89        | Bullish |               7 |
|     66 | BFUSD                                                | BFUSD      | $0.9992       | -0.025%            | -0.000%           | 0.009%        | $0.9993      | $0.9993      | Neutral |               8 |
|     67 | Worldcoin                                            | WLD        | $0.3634       | -9.431%            | -0.079%           | 0.883%        | $0.3634      | $0.3624      | Neutral |               6 |
|     68 | Ethereum Classic                                     | ETC        | $7.20         | -6.499%            | -0.053%           | 0.776%        | $7.22        | $7.21        | Neutral |               6 |
|     69 | Arbitrum                                             | ARB        | $0.1509       | +4.220%            | +0.046%           | 1.563%        | $0.1529      | $0.1526      | Neutral |               6 |
|     70 | Venice Token                                         | VVV        | $21.89        | -8.942%            | -0.066%           | 1.534%        | $21.77       | $21.72       | Bullish |               4 |
|     71 | Lighter                                              | LIT        | $4.26         | -5.546%            | -0.038%           | 1.362%        | $4.19        | $4.15        | Bullish |               9 |
|     72 | POL (ex-MATIC)                                       | POL        | $0.09117      | -1.575%            | -0.011%           | 0.720%        | $0.09153     | $0.0916      | Bearish |               5 |
|     73 | Blockchain Capital                                   | BCAP       | $106.54       | +0.047%            | +0.000%           | 0.004%        | $106.54      | $106.54      | Neutral |               1 |
|     74 | Gate                                                 | GT         | $9.06         | -2.821%            | -0.023%           | 0.469%        | $9.06        | $9.06        | Bearish |               5 |
|     75 | JUST                                                 | JST        | $0.1155       | +12.181%           | +0.097%           | 0.419%        | $0.1156      | $0.1155      | Neutral |               7 |
|     76 | KuCoin                                               | KCS        | $6.79         | -2.256%            | -0.018%           | 0.453%        | $6.80        | $6.80        | Neutral |               7 |
|     77 | Pi Network                                           | PI         | $0.08676      | -10.142%           | -0.087%           | 0.702%        | $0.08687     | $0.08677     | Neutral |               6 |
|     78 | Kaspa                                                | KAS        | $0.03254      | -8.043%            | -0.065%           | 1.030%        | $0.03255     | $0.03259     | Bearish |               6 |
|     79 | Quant                                                | QNT        | $59.95        | -7.677%            | -0.065%           | 0.550%        | $60.37       | $60.50       | Bearish |               9 |
|     80 | Spiko EU T-Bills Money Market Fund                   | EUTBL      | $1.22         | -0.523%            | -0.004%           | 0.043%        | $1.22        | $1.22        | Bullish |               9 |
|     81 | Cosmos Hub                                           | ATOM       | $1.50         | -14.507%           | -0.128%           | 0.746%        | $1.50        | $1.50        | Neutral |               5 |
|     82 | NEXO                                                 | NEXO       | $0.7936       | -4.362%            | -0.036%           | 0.479%        | $0.7978      | $0.7986      | Bearish |               6 |
|     83 | Algorand                                             | ALGO       | $0.08863      | -3.565%            | -0.026%           | 0.950%        | $0.08904     | $0.08895     | Neutral |               7 |
|     84 | Invesco Short Duration US Government Securities Fund | USTB       | $11.21        | +0.048%            | +0.000%           | 0.003%        | $11.21       | $11.21       | Neutral |               3 |
|     85 | Janus Henderson Anemoy AAA CLO Fund                  | JAAA       | $1.05         | +0.064%            | +0.001%           | 0.004%        | $1.05        | $1.05        | Neutral |               2 |
|     86 | ​​Stable                                               | STABLE     | $0.02692      | -9.140%            | -0.075%           | 0.948%        | $0.02695     | $0.0269      | Neutral |               3 |
|     87 | Jupiter                                              | JUP        | $0.2136       | -8.781%            | -0.071%           | 1.008%        | $0.2144      | $0.2142      | Neutral |               8 |
|     88 | GHO                                                  | GHO        | $0.9985       | +0.030%            | +0.000%           | 0.009%        | $0.9986      | $0.9986      | Neutral |               4 |
|     89 | PancakeSwap                                          | CAKE       | $2.20         | +2.571%            | +0.024%           | 0.706%        | $2.20        | $2.20        | Neutral |               8 |
|     90 | Dash                                                 | DASH       | $52.16        | -6.517%            | -0.051%           | 1.055%        | $51.38       | $50.95       | Bullish |               7 |
|     91 | Render                                               | RENDER     | $1.31         | -7.119%            | -0.059%           | 0.742%        | $1.31        | $1.31        | Neutral |               6 |
|     92 | Janus Henderson Anemoy Treasury Fund                 | JTRSY      | $1.12         | +0.030%            | +0.000%           | 0.002%        | $1.12        | $1.12        | Neutral |               2 |
|     93 | Filecoin                                             | FIL        | $0.8077       | +1.991%            | +0.026%           | 1.413%        | $0.8136      | $0.8137      | Bearish |               7 |
|     94 | Akedo                                                | AKE        | $0.02852      | +100.831%          | +0.649%           | 3.515%        | $0.02824     | $0.02818     | Bullish |               8 |
|     95 | Beldex                                               | BDX        | $0.07563      | -1.493%            | -0.012%           | 0.164%        | $0.07558     | $0.07558     | Bullish |               6 |
|     96 | VeChain                                              | VET        | $0.007046     | -6.503%            | -0.051%           | 1.045%        | $0.007077    | $0.007082    | Bearish |               9 |
|     97 | Ether.fi                                             | ETHFI      | $0.5858       | -13.343%           | -0.106%           | 1.595%        | $0.5877      | $0.5878      | Bearish |               9 |
|     98 | Usual USD                                            | USD0       | $0.9991       | +0.053%            | +0.000%           | 0.020%        | $0.9982      | $0.998       | Bullish |               2 |
|     99 | XDC Network                                          | XDC        | $0.02746      | -2.844%            | -0.023%           | 0.538%        | $0.02749     | $0.0275      | Bearish |              10 |
|    100 | Flare                                                | FLR        | $0.006264     | -0.929%            | -0.005%           | 0.679%        | $0.006261    | $0.006258    | Bullish |               9 |

## 7. Figures

Figures were not generated for this run.

## 8. Data quality

| metric                     | value                |
|:---------------------------|:---------------------|
| input_rows                 | 12064                |
| coins_input                | 100                  |
| invalid_rows_dropped       | 0                    |
| invalid_prices_set_missing | 0                    |
| duplicates_removed         | 0                    |
| rows_outside_window        | 0                    |
| prices_interpolated        | 0                    |
| rows_missing_price_dropped | 0                    |
| coins_excluded             | none                 |
| output_rows                | 12064                |
| coins_output               | 100                  |
| median_interval_minutes    | 60.0                 |
| window_start_utc           | 2026-09-11 07:00 UTC |
| window_end_utc             | 2026-09-16 09:53 UTC |

## 9. Methodology

* **Periodic return:** r_t = (P_t − P_(t−1)) / P_(t−1) × 100, computed at the original timestamp resolution.
* **5-day return:** (P_last − P_first) / P_first × 100 over the window.
* **Moving averages:** MA(k)_t = mean of the k most recent prices including t (k = 3, 5 periods).
* **Trend:** Bullish if Price > MA(3) > MA(5); Bearish if Price < MA(3) < MA(5);
  otherwise Neutral.
* **Volatility:** population standard deviation of returns, σ = √(Σ(x − μ)² / N). Returns (not prices) are used
  so that coins with very different price levels are comparable.
* **Outliers:** Z = (x − μ) / σ per coin; |Z| > 2 flagged, |Z| > 3 extreme.
* **Unusual volatility:** σ above the Tukey fence Q3 + 1.5 × IQR of all coins' σ.
* **24/7 markets:** no business-day calendar; every hour, night and weekend is a valid observation.

## 10. Limitations

* Five days of hourly data is a short sample; statistics are noisy and regime-dependent.
* MA(3)/MA(5) on hourly data describe hour-scale momentum only.
* Z-scores use each coin's own mean/σ, which include the outliers themselves (masking effect), and assume a
  roughly stable distribution; crypto returns are fat-tailed, so more |Z| > 2 events than the normal
  baseline are expected.
* CoinGecko prices are aggregated across exchanges and may differ from any single venue; the last point of each
  series may cover a partial interval.
* The market-cap snapshot and the historical series are retrieved seconds to minutes apart.
