# Data sources

The notebook builds its dataset on the fly from two public APIs. None of the
raw data is committed to this repo. Reproducing the analysis requires:

## 1. World Bank Open Data API

No API key needed. Indicator codes used by the notebook (cell 8):

| Indicator | World Bank code | Description |
|-----------|----------------|-------------|
| GDP growth (annual %) | `NY.GDP.MKTP.KD.ZG` | Real GDP growth |
| GDP per capita (constant 2015 USD) | `NY.GDP.PCAP.KD` | Income proxy for recovery rates |
| Inflation, consumer prices (annual %) | `FP.CPI.TOTL.ZG` | Headline CPI |
| Unemployment rate (% of total labor force) | `SL.UEM.TOTL.ZS` | Labor market stress |
| Current account balance (% of GDP) | `BN.CAB.XOKA.GD.ZS` | External flow imbalance |
| Total reserves (months of imports) | `FI.RES.TOTL.MO` | Liquidity cover |
| Trade (% of GDP) | `NE.TRD.GNFS.ZS` | Trade openness |
| FDI net inflows (% of GDP) | `BX.KLT.DINV.WD.GD.ZS` | Investor sentiment |
| External debt stocks (% of GNI) | `DT.DOD.DECT.GN.ZS` | Debt burden |
| Debt service (% of exports) | `DT.TDS.DECT.EX.ZS` | Debt service cost |
| Central government debt (% of GDP) | `GC.DOD.TOTL.GD.ZS` | Public debt |
| Government revenue (% of GDP) | `GC.REV.XGRT.GD.ZS` | Fiscal capacity |
| Government expenditure (% of GDP) | `GC.XPN.TOTL.GD.ZS` | Fiscal spending |
| Broad money (% of GDP) | `FM.LBL.BMNY.GD.ZS` | Monetary depth |
| Domestic credit to private sector (% of GDP) | `FS.AST.PRVT.GD.ZS` | Credit depth |

Endpoint: `https://api.worldbank.org/v2/country/<codes>/indicator/<code>?format=json&date=1990:2023&per_page=2000`

## 2. FRED (Federal Reserve Economic Data) API

Requires a free API key. Get one at https://fred.stlouisfed.org/docs/api/api_key.html.
Set it as an environment variable before running the notebook:

    export FRED_API_KEY=<your_key>

The notebook reads it via `os.environ["FRED_API_KEY"]` (the prior hard-coded
placeholder `"----"` has been removed).

Series IDs used:

| Series | FRED ID |
|--------|---------|
| VIX (annual average) | `VIXCLS` |
| US 10-year Treasury yield | `DGS10` |
| USD broad index | `DTWEXBGS` |
| High yield credit spread | `BAMLH0A0HYM2` |
| TED spread | `TEDRATE` (discontinued; substitute with `T10Y2Y` from 2022 onwards) |
| Yield curve slope (10Y - 2Y) | `T10Y2Y` |

## 3. Sovereign default events

The default-event database in notebook cell 6 is hand-curated and cross-
referenced against four sources. Definition of a default event includes
missed payments, debt restructuring with haircuts, IMF bailouts with debt
relief, and selective default ratings. 88 events across 63 countries
(1989-2022). See cell 6 for the literal `DEFAULT_EVENTS` dict.

Source-document citations:

- Reinhart, C. M., & Rogoff, K. S. (2009). *This Time is Different: Eight Centuries of Financial Folly.* Princeton University Press.
- S&P Global Ratings. *Sovereign Defaults and Rating Transition Data* (annual studies).
- Moody's Investors Service. *Sovereign Default and Recovery Rates* (annual studies).
- Bank of Canada / Bank of England joint database: <https://www.bankofcanada.ca/wp-content/uploads/2014/02/technical-report-101.pdf>

## 4. Caveats from the README's Limitations section

- 56 of the 88 default events are stamped in the 1990s, many of them
  carryover from the 1980s debt crisis. The cleaner subset is the 32
  post-2000 events.
- World Bank indicators have missing values for some country-years
  (former Soviet states pre-independence, small economies with sparse
  reporting). The notebook imputes train medians after the temporal
  split.
