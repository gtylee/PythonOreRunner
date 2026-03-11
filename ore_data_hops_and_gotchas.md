# ORE Data Hops and Gotchas

This note is for the practical problem of creating or extending ORE market data without getting lost in the naming conventions and missing-data traps.

It is based on the source, not just the examples. The key point is that ORE's market-data flow is mostly string-driven. Very little is inferred. Most failures happen because two files use slightly different names for the same thing, or because a quote is "obviously implied" to a human but not actually requested anywhere in the config graph.

## 1. The actual hop order

The runtime wiring starts in `ore.xml`:

1. `ore.xml` points to `marketDataFile`, `fixingDataFile`, `curveConfigFile`, `conventionsFile`, and `marketConfigFile`.
2. `OREApp::buildCsvLoader()` reads the market, fixing, and dividend files from `inputPath`.
3. `TodaysMarketParameters` reads `todaysmarket.xml` and turns user-facing names like `EUR-EURIBOR-6M` or `EURUSD` into curve specs like `Yield/EUR/EUR6M` or `FX/EUR/USD`.
4. `CurveConfigurations` parses the referenced curve specs from `curveconfig.xml`.
5. `TodaysMarket` builds a dependency graph, loads fixings and dividends first, pulls all FX spot quotes into triangulation, and then builds the requested curves/configurations.

Source pointers:

- `OREAnalytics/orea/app/oreapp.cpp:188-231`
- `OREData/ored/marketdata/todaysmarketparameters.cpp:153-295`
- `OREData/ored/configuration/curveconfigurations.cpp:220-294`
- `OREData/ored/marketdata/todaysmarket.cpp:103-220`

## 2. What each file is really responsible for

### `marketdata.csv` or `market.txt`

This is the raw quote store. The CSV loader expects:

- market and fixing lines: `date, quote_id, value`
- dividend lines: `ex_date, name, amount[, pay_date[, announcement_date]]`

Important behavior from the loader:

- blank lines and `#` comments are skipped
- quote IDs are parsed purely from the string
- duplicates are silently skipped
- FX spot duplicates are special: if both `FX/RATE/EUR/USD` and `FX/RATE/USD/EUR` are present, ORE keeps the dominant pair and may replace the other
- today's fixings are ignored unless `implyTodaysFixings = Y`

Source pointers:

- `OREData/ored/marketdata/csvloader.cpp:100-189`
- `OREData/ored/marketdata/loader.cpp:69-98`

### `todaysmarket.xml`

This file does not define quotes. It defines the mapping from business names to curve specs and assigns those mappings to market configurations such as `default`, `simulation`, `sensitivity`, etc.

Examples:

- `DiscountingCurve currency="EUR"` -> `Yield/EUR/EUR1D`
- `Index name="EUR-EURIBOR-6M"` -> `Yield/EUR/EUR6M`
- `FxSpot pair="EURUSD"` -> `FX/EUR/USD`

Important behavior:

- the `default` configuration is auto-filled with default IDs for every market object type, but not with actual entries
- duplicate entries inside a node are rejected
- overlap of names between `YieldCurves` and `IndexForwardingCurves` under the same ID is rejected
- swap indices are handled differently from the rest

Source pointers:

- `OREData/ored/marketdata/todaysmarketparameters.cpp:126-140`
- `OREData/ored/marketdata/todaysmarketparameters.cpp:153-218`
- `OREData/ored/marketdata/todaysmarketparameters.cpp:285-295`
- `OREData/ored/marketdata/todaysmarketparameters.cpp:298-330`

### `curveconfig.xml`

This file tells ORE how to build each curve and which exact quote strings and convention IDs it expects.

This is where the hidden strictness lives:

- segment names must match the allowed node names exactly, e.g. `Simple`, `CrossCurrency`, `DiscountRatio`
- segment `Type` must be valid for that segment node
- quotes are not discovered from the market file; they are requested explicitly from here
- convention IDs are also pulled from here

For yield curves specifically:

- every segment contributes exact quote IDs
- cross-currency segments automatically add the FX spot quote and also the inverted FX pair because the loader may have removed one side during FX dominance resolution

Source pointers:

- `OREData/ored/configuration/yieldcurveconfig.cpp:191-214`
- `OREData/ored/configuration/yieldcurveconfig.cpp:216-308`
- `OREData/ored/configuration/yieldcurveconfig.cpp:363-410`

### `conventions.xml`

This file is the final piece that turns generic quote families into instruments/helpers.

If a curve segment says `Conventions = USD-LIBOR-CONVENTIONS`, that ID must exist here. ORE can use dated conventions, and if the exact evaluation date is missing it will fall back to the nearest earlier dated block with a warning.

Source pointer:

- `OREData/ored/configuration/conventions.cpp:74-100`

## 3. The matching rules that bite people

These strings must line up exactly:

- quote IDs in `marketdata.csv`
- quote IDs generated or referenced by `curveconfig.xml`
- curve specs used in `todaysmarket.xml`
- convention IDs used in `curveconfig.xml`
- market object names in `todaysmarket.xml`
- portfolio indices and names that later request these market objects

Typical examples:

- `todaysmarket.xml` maps `Index name="EUR-EURIBOR-6M"` to `Yield/EUR/EUR6M`
- `curveconfig.xml` must contain a yield curve with `CurveId = EUR6M`
- its segments must request quotes that exist in `marketdata.csv`
- its `Conventions` IDs must exist in `conventions.xml`

If any one of those links is broken, ORE does not "best effort" its way through it. It usually throws late, after building a dependency graph, which makes the root cause feel farther away than it is.

## 4. Missing-data behavior: where ORE is strict vs forgiving

### Strict

- mandatory quotes: `Loader::get()` throws if the quote is not found
- unrecognized curve segment node names or invalid type/node combinations
- missing required convention IDs
- duplicate or inconsistent `todaysmarket.xml` mappings
- overlapping names between yield curves and index curves in the same market config

### Forgiving

- optional quotes can be marked with `optional="true"` on `<Quote>`
- convention blocks can fall back to an older dated convention set
- if both FX spot directions exist, one may be dropped automatically
- some curve-config parsing in `CurveConfigurations::minimalCurveConfig()` is swallowed internally while building the minimal set

Source pointers:

- `OREData/ored/marketdata/loader.cpp:69-81`
- `OREData/ored/configuration/yieldcurveconfig.cpp:398-403`
- `OREData/ored/configuration/conventions.cpp:81-94`
- `OREData/ored/configuration/curveconfigurations.cpp:233-239`

## 5. The real gotchas

### Gotcha 1: `todaysmarket.xml` does not validate that the target curve spec exists

You can happily map `EUR-EURIBOR-6M -> Yield/EUR/EUR6M`, but if `EUR6M` is missing from `curveconfig.xml`, the failure only appears later when the market is being built.

### Gotcha 2: quote names are opaque strings

The quote parser understands many shapes, but there is no central schema check saying "these are all quotes required by this run and here is the exact diff versus your market file" before build starts.

### Gotcha 3: FX pairs are slippery by design

The loader can remove one of `EUR/USD` or `USD/EUR` based on dominance. Yield curve config works around this for cross-currency segments by asking for both sides, but that logic is local to those configs. If you hand-maintain FX data, it is easy to think you have two equivalent quotes when ORE only keeps one.

### Gotcha 4: default configuration inheritance is partial

`TodaysMarketParameters` auto-fills market object IDs for `default`, but it does not invent the underlying mappings. A missing `<YieldCurves id="default">` block is still a missing block.

### Gotcha 5: yield curve names and index curve names cannot overlap

This is easy to hit when teams try to simplify naming and call both the forwarding index and the discount curve `EUR`.

### Gotcha 6: fixings behavior is easy to misread

Today's fixing is ignored unless `implyTodaysFixings` is enabled. That matters when a build helper needs the fixing on the evaluation date.

### Gotcha 7: errors often surface at build time, not declaration time

The dependency graph is assembled first and then objects are topologically built. A broken quote, convention, or referenced curve often shows up only when the specific node is reached.

## 6. A practical workflow for creating new data

When adding a new currency/index/curve, work in this order:

1. Start in `curveconfig.xml`.
   Decide the exact `CurveId`, segment type, quote strings, and convention IDs.
2. Create the needed convention IDs in `conventions.xml`.
   Do this before market data so the helper definitions are fixed.
3. Add the raw quotes to `marketdata.csv`.
   Use the exact quote strings from the curve config, not a human-friendly variation.
4. Add the mapping in `todaysmarket.xml`.
   Map the business-facing name used by trades or engines to the curve spec.
5. Wire the configuration in `ore.xml`.
   Make sure the requested analytic uses the intended market configuration.
6. Run once and inspect the first missing item, not the full stack trace.
   In most cases the first broken string match is the real issue.

## 7. What should be fixed in ORE to make this less slippery

These are the concrete improvements suggested by the code:

### 1. Add a preflight "required quotes / missing quotes / unused quotes" report

The code already knows how to derive required quotes from `TodaysMarketParameters + CurveConfigurations` via `CurveConfigurations::quotes(...)`. That should be exposed as a first-class validation step before curve building.

Relevant source:

- `OREData/ored/configuration/curveconfigurations.cpp:245-267`

### 2. Add a preflight "required conventions / missing conventions" report

Same story for conventions. The engine can already derive the convention IDs it needs. This should fail early with a small report.

Relevant source:

- `OREData/ored/configuration/curveconfigurations.cpp:282-318`

### 3. Improve error locality between `todaysmarket.xml` and `curveconfig.xml`

Right now a bad mapping in `todaysmarket.xml` usually fails later when `parseCurveSpec()` or `get()` is reached indirectly. A direct validation pass that checks every mapped spec against `CurveConfigurations` would save a lot of time.

### 4. Expose FX dominance decisions clearly

The logic is correct enough, but operationally opaque. A dedicated summary saying "kept `FX/RATE/EUR/USD`, removed `FX/RATE/USD/EUR`" would help.

Relevant source:

- `OREData/ored/marketdata/csvloader.cpp:141-156`
- `OREData/ored/marketdata/loader.cpp:85-97`

### 5. Tighten CSV dividend validation

There is a suspicious validation check in `CSVLoader`:

```cpp
QL_REQUIRE(tokens.size() >= 3 || tokens.size() <= 5, ...)
```

That condition is effectively always true, so the intended `3..5` dividend token validation is not actually enforced. It looks like this should be `&&`.

Relevant source:

- `OREData/ored/marketdata/csvloader.cpp:121-125`

### 6. Add one canonical naming guide for quote IDs

The parser accepts a wide range of instrument types, but the project does not have a short operational guide that says:

- here is the quote naming pattern by asset class
- here is where the same string must reappear
- here is which names are user-facing aliases versus internal specs

That gap is the main reason the data setup feels slippery.

## 8. Short checklist before running

- Does every `Conventions` ID in `curveconfig.xml` exist in `conventions.xml`?
- Does every quote string in `curveconfig.xml` exist in `marketdata.csv` for the `asofDate`?
- Does every curve spec in `todaysmarket.xml` point to a real curve config?
- Are yield curve names and index curve names distinct inside the same `todaysmarket` ID?
- If cross-currency is involved, do you understand which FX direction ORE will keep?
- If today's fixing matters, is `implyTodaysFixings` set correctly?
- Are you editing the right market configuration (`default`, `simulation`, `sensitivity`, etc.)?

## 9. Bottom line

ORE market setup is not mainly hard because the math is obscure. It is hard because the data model is split across four files and the links are mostly just strings:

- raw quote ID
- convention ID
- curve ID
- curve spec
- market-object alias
- configuration ID

Once you treat it as a graph of exact string contracts, the behavior becomes much more predictable.
