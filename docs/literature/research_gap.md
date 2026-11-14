# Research-gap check (T1.12)

**Claim in plan v2 (§4):** no study applies and compares GA, BPSO and BGWO feature
selection on a future-elite talent label with player-grouped temporal splits.

**Verdict: confirmed, with one nearby study to cite.** PSO has been used for feature
selection on FIFA data, but for market-value regression (Behravan and Razavi, 2021),
not for talent identification and not against other metaheuristics. The claim stays,
worded narrowly.

## Searches (14 Nov 2026)

Run with a general web search engine and the Crossref and Semantic Scholar APIs.
Google Scholar blocks automated queries, so the last step below repeats the key
searches there by hand.

| Query | What came up | Close? |
| --- | --- | --- |
| "feature selection" "genetic algorithm" football talent identification young players | a GA wrapper on heart-rate data of 44 AFL players (Australian rules, n = 44); a GA feature-selection thesis for NFL draft prospects; reviews of ML in academy football | No: other sports, not FIFA data, no comparison of optimizers |
| "particle swarm" FIFA players dataset feature selection | Behravan and Razavi (2021): PSO feature selection plus SVR for transfer value on FIFA 20 | Partly: same optimizer family and data source, different task (value regression) |
| "grey wolf" optimizer soccer football player prediction feature selection | Boruta for market value; chi-square selection for position classification | No |
| FIFA video game dataset predict player potential future overall rating | Feldman (2025); several student projects classifying "high potential" from current attributes | No metaheuristics; Feldman is the closest talent study |
| FIFA players classification, nine feature selection algorithms, three balancing techniques | Al-Asadi and Tasdemir (2021), *IEEE Access*: filter / wrapper methods for position prediction | No: position, not talent; no GA / PSO / GWO comparison |
| football talent identification metaheuristic feature selection, GA / PSO / GWO, youth, future elite | Tang et al. (2026) systematic review (27 studies, none with metaheuristic selection); youth-selection studies with RFE, Lasso, Boruta | No |

## Gap statement for the paper

> Metaheuristic wrapper feature selection is well studied on benchmark datasets, and
> FIFA / EA FC ratings are widely used for rating, position and market-value prediction;
> one study used PSO to select features for transfer-value regression on FIFA 20 data.
> To our knowledge, no study has compared a Genetic Algorithm, Binary PSO and Binary
> Grey Wolf Optimizer for identifying future elite players, nor evaluated such selection
> under a temporal, player-grouped split with an untouched test season.

## Still to do by hand

- [ ] Repeat the first three queries on Google Scholar (two pages each) and add anything close to this file.
