# TODOS

**State of the system:** Modularity Roadmap Steps 1–5 complete. ETL lives in `etl.py`; API can start without scrapers when `ILGA_LOAD_ONLY=1`. Scorecards and Moneyball are cached to `cache/` and reused when member data is unchanged. GraphQL resolvers use request-scoped batch loaders (scorecard, moneyball profile, bill, member). `ILGA_PROFILE=dev|prod`; config; seating; SSR advocacy at `/advocacy`; Moneyball v2 (shell bill filter, institutional bonus, chair-first Power Broker). **Unified GraphQL `search` query** — free-text search across members, bills, and committees with relevance scoring, entity-type filtering, and pagination (`search.py`). **Committee Power Dashboard** — each advocacy card now shows committee assignments with leadership roles, power explanations, and per-committee bill advancement stats. **Institutional Power Badges** — visual hierarchy badges (LEADERSHIP, COMMITTEE CHAIR, TOP 5% INFLUENCE) at the top of each advocacy card with click-to-expand explanations. **Power Card Redesign** — advocacy cards restructured into a consolidated "Power Card" layout matching advocacy professional needs: badge row, inline name with party/district, "Why this person matters" power context box, compact scorecard with caucus comparison, contact row (phone/email/ILGA profile), and expandable detail sections. **True Influence Engine** — new `influence.py` module with unified influence scoring: betweenness centrality (bridge influence), vote pivotality (swing voter power), sponsor pull (ML-predicted bill success), and InfluenceScore (0–100 composite). Intelligence dashboard Influence tab. GraphQL `influence_leaderboard` query. Coalition influence enrichment (top influencer + bridge member per bloc). **Prediction table v5** — fixed critical bugs: vetoed/dead bills no longer predicted as ADVANCE; added lifecycle status (OPEN/PASSED/VETOED/DEAD) with visual distinction; staleness features for the model; sortable table columns. **ML Pipeline Bug Audit** — fixed TF-IDF zero-vector fallback, KeyError guards, dynamic majority party computation, sponsor rate null guard, vote tie handling, anomaly NaN guard, defensive date-parse logging. **Intelligence Story Redesign** — new narrative-driven `/intelligence` executive summary (Bills to Watch, Power Movers, Coalition Landscape, Anomaly Alerts), deep-dive pages at `/intelligence/member/:id` and `/intelligence/bill/:id`, raw data tables moved to `/intelligence/raw`. **Legislative Power Map** — interactive D3.js force-directed graph visualization at `/explore` showing all 180 legislators as connected nodes: sized by influence, colored by party, linked by co-sponsorship. Topic filter narrows to relevant committees; ZIP input highlights your legislators; click-to-open detail panel shows stats and influence signals.

---

## Current

- **CI: install ML extras so tests can import numpy/polars (2026-02-14):**
  - GitHub Actions was failing with `ModuleNotFoundError: No module named 'numpy'` (in `test_action_classifier.py` via `ilga_graph.ml.features`) and `No module named 'polars'` (in `test_panel_labels.py`). CI only installed `.[dev]`, so `.[ml]` (polars, scikit-learn/numpy, etc.) was missing.
  - **Fix:** `.github/workflows/ci.yml` — changed install step to `pip install -e ".[dev,ml]"` so ML-dependent tests run with required deps.

- **Lint fixes (2026-02-14):**
  - `src/ilga_graph/ml/features.py`: Defined missing `ft_tfidf_id_to_idx` in both `build_feature_matrix()` and panel path (same row order as `tfidf_bill_ids`) so full-text TF-IDF lookup is in scope when `_ft_has_features` is true.
  - `src/ilga_graph/ml/member_value.py`: Removed unused `tier_name` assignment in topic-coalition tier loop (F841).

- **Member Value Model — Undervalued/Overvalued Detection + Issue Recruiter (2026-02-14):**
  - **Problem:** Moneyball (0-100) and Influence (0-100) scores are handcrafted weighted composites. They identify who IS effective, but not who COULD BE effective. Advocacy teams need to know: "Which legislators have the structural position and topic alignment to move the needle on my issue, but aren't on anyone's radar?"
  - **Solution — ML-based value model:** Train a Ridge Regression model (with Leave-One-Out CV for robustness on N~180) to predict legislative effectiveness from structural features, then compare predicted vs. actual to surface undervalued members. Combine with topic coalition data for issue-specific recruitment rankings.
  - **New module — `src/ilga_graph/ml/member_value.py`:**
    - `MemberValueProfile` dataclass: predicted_effectiveness, actual_effectiveness, value_residual, value_percentile, value_label (Undervalued/Fairly Valued/Overvalued), top_recruitment_topics.
    - `TopicRecruitmentScore` dataclass: per-member-per-topic composite of affinity (35%), predicted effectiveness (30%), persuadability (20%), network reach (15%).
    - `build_member_feature_matrix()`: Assembles features from Node2Vec embeddings (PCA-reduced to 8 dims), Moneyball network metrics (centrality, betweenness, collaborators), demographics (party, chamber, career tenure), institutional role (leadership, committee chair), topic YES rates. ~33 features for 180 members.
    - `train_value_model()`: Ridge regression with LOO-CV. Clips predictions to [0,1]. Reports R-squared and MAE.
    - `compute_value_scores()`: Orchestrator — builds features, trains model, computes residuals, assigns labels via percentile thresholds.
    - `compute_recruitment_rankings()`: Per-topic scoring. Swing-tier members with high predicted effectiveness rank highest — they're the "gettable and capable" targets.
    - `run_member_value_pipeline()`: Single entry point for the ML pipeline step.
  - **Pipeline integration — `scripts/ml_run.py`:**
    - New Step 6 (after Coalition Discovery, before Anomaly Detection). Pipeline is now 9 steps (0-8).
    - Loads Moneyball from `cache/moneyball.json`, embeddings, dim_members, topic_coalitions.
    - Saves `processed/member_value_scores.parquet` and `processed/member_recruitment.json`.
  - **ML Loader — `src/ilga_graph/ml_loader.py`:**
    - New `MemberValueScore` dataclass. `MLData` extended with `member_value_scores`, `topic_recruitment`, `member_value_meta`.
    - Loaded at startup alongside other ML artifacts.
  - **API routes — `src/ilga_graph/main.py`:**
    - `GET /intelligence/recruitment` — full recruitment page with value leaderboard + topic selector.
    - `GET /intelligence/recruitment/{topic}` — HTMX partial with per-topic ranked member list.
    - Extended `GET /intelligence/member/{member_id}` — "Value Assessment" card with predicted vs actual effectiveness, residual, value label, and top recruitment topics.
  - **Templates:**
    - `intelligence_recruitment.html` (NEW): Value leaderboard table (filterable by value label/chamber), topic selector with HTMX-loaded per-topic results.
    - `_recruitment_topic_partial.html` (NEW): Per-topic table with recruitment score breakdown (affinity, effectiveness, persuadability, network), coalition tier badge, value label.
    - `intelligence_member.html`: New "Value Assessment" section between Moneyball and Notable Bills.
    - `intelligence_summary.html`: Added "Issue Recruiter" nav link.
    - `base.html`: Added value badge CSS (.value-undervalued, .value-fairly-valued, .value-overvalued).
  - **Design decisions:**
    - Ridge Regression over GradientBoosting: N=180 would overfit with tree ensembles. Ridge + LOO-CV provides robust estimates.
    - PCA on embeddings: Reduces 64-dim Node2Vec to 8 components before combining with ~25 tabular features. Prevents curse of dimensionality.
    - Persuadability weighting: Swing=1.0, Lean Support=0.7, Lean Oppose=0.5, Champion=0.2, Oppose=0.1. Champions rank low because they're already on your side.
    - Target variable is effectiveness_rate (laws_passed/laws_filed) from Moneyball — the ground truth outcome.
  - **Next steps:** Run `make ml-run` to generate value scores and recruitment rankings. Future phases: Member-Bill Recommender (collaborative filtering), cross-session prediction, vote prediction.

- **Full-text leakage fix — advance model (2026-02-14):**
  - **Problem:** The advance model's top feature importance was dominated by `ft_tfidf_*` features — specifically session-year proxies like "2025" and post-hoc bill-text artifacts ("enrolled", LRB numbers). These leaked outcome information: a bill whose full text says "enrolled" has obviously already passed. The Forecast model already excluded full-text, but the advance model still used it, getting ~50% of its feature importance from full-text TF-IDF.
  - **Fix:** Excluded full-text TF-IDF and full-text-derived content metadata from **all** model modes (advance + forecast), not just forecast.
  - **Changes:**
    - `src/ilga_graph/ml/features.py`: New constant `FULLTEXT_DROP_COLUMNS` (7 content-metadata columns). Applied unconditionally in both `build_feature_matrix()` and `build_panel_feature_matrix()`. Full-text TF-IDF matrix is now always zero-width (`csr_matrix((n, 0))`), so `ft_tfidf_*` features are never generated.
    - `build_full_text_features()` still exists (no dead code removed) but is never called during matrix construction.
    - Full-text PDFs remain in `cache/bills.json` for search, display, and future non-ML use.
  - **Impact:** After re-running `make ml-run`, the advance model will rely on genuine predictive features (sponsor history, committee dynamics, slip patterns, temporal signals, graph embeddings) instead of text artifacts. SHAP explanations will become more meaningful.
  - **Next step:** Run `make ml-run` to retrain with the corrected feature set.

- **Bill-to-law process in reference model (2026-02-14):**
  - **Added:** Canonical "How does a bill become law in Illinois?" 6-step overview into the reference model so the codebase and UI use one source of truth.
  - **reference/ilga_rules.json:** New top-level key `bill_to_law_process` — array of 6 steps (Introduction of Bill; Committee Work — Hearings; Committee Work — Markup, Amendments, Report; Floor Debate; Passage and Consideration in Second Chamber; Gubernatorial Action). Each step has `step`, `title`, `body`. Aligns with existing `stages` (FILED → SIGNED/VETOED).
  - **rule_engine.py:** `get_bill_to_law_process()` returns the list from the glossary (cached).
  - **main.py:** `intelligence_bill_detail` loads and passes `bill_to_law_process` to the template (both found and not-found responses).
  - **intelligence_bill.html:** Collapsible `<details>` section "How does a bill become law in Illinois?" below Pipeline Progress, rendering the 6 steps from the model.

- **Bill page: accurate last action date + full actions table (2026-02-14):**
  - **Problem:** Bill SB1531 (and others) showed last action date as June 1, 2025 when the actual last action was Sep 17 — the UI was using the ML score's `last_action_date`, which is frozen at pipeline run time; the cache (`bills.json`) already had the correct later actions in the array.
  - **Fix:** (1) **state.bills_lookup** — set in lifespan so bill detail can resolve bill by `leg_id` (ML score `bill_id`). (2) **Last action from action_history** — when the bill page has `action_history` from the live cache, we sort actions by date, take the chronologically last one, and set `last_action_date`, `last_action_text`, and `days_since_action` from it (overriding the score). (3) **Fallbacks** — if `bills_lookup.get(bill_id)` misses, try `bill_lookup.get(bill_number)`; then if still None, scan `bills_lookup` for any bill with the same `bill_number` and pick the one with the **latest action date** so we use the most up-to-date copy from the cache. (4) **Full actions table** — added "All actions (table)" with columns Date, Chamber, Action, Category, Signal; Action History section now shows timeline + table and "Last action: &lt;date&gt; — &lt;text&gt;" at the top.
  - **Files:** `src/ilga_graph/main.py` (`_parse_action_date`, bill detail action_history + override + latest-action fallback), `src/ilga_graph/templates/intelligence_bill.html` (table + last-action note), `src/ilga_graph/templates/base.html` (`.bill-actions-table` CSS).

- **Pipeline stage rollback fix — HB3356 / Rule 19(b) (2026-02-14):**
  - **Problem:** HB3356 (Hair Braiding Licensure Repeal) passed both House and Senate but was then re-referred to Rules Committee (Rule 19(b)) on 7/01/2025 before concurrence. The tool incorrectly showed "Sent to Governor" because we used the *highest* stage ever reached instead of the *current* stage after the last action.
  - **Root cause:** `bill_outcome_from_actions()` in `action_classifier.py` only tracked `highest_stage`. When the chronologically last action was "Rule 19(b) / Re-referred to Rules Committee", we never downgraded the stage, so the UI still showed Governor.
  - **Fix:** (1) **Rollback detection** — `_is_stage_rollback()` treats Rule 19(a)/(b), Rule 3-9(a)/(b), and "Re-referred to Rules Committee/Assignments" as rollbacks. (2) **Current stage** — we now track `current_stage` in addition to `highest_stage`; when a rollback action is seen and current_stage is CROSSED_CHAMBERS/PASSED_BOTH/GOVERNOR, we set current_stage = IN_COMMITTEE. (3) **Stage display** — `compute_bill_stage()` in `features.py` uses `current_stage` (with fallback to `highest_stage` for backward compat). (4) **Chronological order** — `score_all_bills()` in `bill_predictor.py` builds `action_map` from `df_actions.sort("date")` so rollbacks are applied in correct order.
  - **Files:** `src/ilga_graph/ml/action_classifier.py`, `src/ilga_graph/ml/features.py`, `src/ilga_graph/ml/bill_predictor.py`. **Tests:** `tests/test_action_classifier.py` — HB3356-style action list asserts current_stage is IN_COMMITTEE and compute_bill_stage returns IN_COMMITTEE.

- **SHAP Prediction Explanations — "Why This Score?" (2026-02-14):**
  - **Problem:** The GradientBoostingClassifier outputs a raw probability, but users have no way to understand *why* a specific bill received its score. Legislators and advocacy teams need actionable insight — which factors are pushing a bill's chances up, and which are dragging them down.
  - **Solution — SHAP (Shapley Additive exPlanations):** Integrated `shap.TreeExplainer` to compute per-bill feature contributions, converting log-odds SHAP values to probability-space percentage impacts via `scipy.special.expit`.
  - **Files changed:**
    - `pyproject.toml`: Added `shap` and `scipy` to `[ml]` dependencies.
    - `src/ilga_graph/ml/features.py`: Added `FEATURE_NAME_MAPPING` dictionary (40+ raw feature names → human-readable labels), `CATEGORICAL_PREFIXES` list, and `humanize_feature_name()` helper with smart fallbacks for TF-IDF, embedding, and unknown features.
    - `src/ilga_graph/ml/bill_predictor.py`: In `run_auto()`, saves the raw (pre-calibration) `GradientBoostingClassifier` to `bill_predictor_raw.pkl`, and saves the scoring feature matrix + metadata to `shap_feature_matrix.npz` and `shap_feature_meta.json`.
    - `src/ilga_graph/ml/explainer.py` **(NEW)**: `SHAPExplainer` class — initialised once at startup with `TreeExplainer`, caches the explainer object. `explain_prediction()` computes SHAP values, converts log-odds → probability impacts, groups one-hot encoded categoricals (`sponsor_party_democrat` + `sponsor_party_republican` → `sponsor_party`), ranks by absolute magnitude, returns top 3 positive and top 3 negative factors with formatted strings.
    - `src/ilga_graph/ml_loader.py`: `MLData` extended with `explainer`, `feature_matrix`, `feature_bill_ids`, `feature_names`, `_bill_id_to_row`. `load_ml_data()` loads SHAP artifacts and initialises the explainer at startup (graceful degradation if artifacts missing).
    - `src/ilga_graph/main.py`: Added `PredictionFactor` and `PredictionExplanation` Strawberry GraphQL types. Added `prediction_explanation(bill_id)` resolver to Query. Added `GET /api/bills/{bill_id}/explanation` REST endpoint returning an HTML fragment for HTMX lazy-loading.
    - `src/ilga_graph/templates/_explanation_partial.html` **(NEW)**: HTMX fragment with green badges (positive factors) and red badges (negative factors), base probability note, and graceful fallback messages.
    - `src/ilga_graph/templates/intelligence_bill.html`: Added `hx-get` div after the prediction card that lazy-loads the explanation fragment on page load.
  - **Architecture:** TreeExplainer is initialised once at startup (not per-request). Feature matrix is stored in memory as a sparse matrix. SHAP values are computed on-demand per bill via the `/api/bills/{id}/explanation` endpoint to avoid blocking bulk bill listing. CalibratedClassifierCV is not compatible with TreeExplainer, so the raw model is saved separately.
  - **Next steps:** Run `make ml-run` to generate SHAP artifacts (`bill_predictor_raw.pkl`, `shap_feature_matrix.npz`, `shap_feature_meta.json`), then `make dev` to verify explanation badges appear on bill detail pages. Consider adding SHAP waterfall/beeswarm charts as a future enhancement.

- **Unified Run Log & Dashboard (2026-02-14):**
  - **Problem:** Scraping and startup had timing logs (e.g. `.startup_timings.csv`); ML pipeline had none. No single place to see bottlenecks and "how things are looking" across scrape, ml_run, and startup.
  - **Solution:** Append-only run log (`.run_log.jsonl`) plus terminal and web dashboards.
  - **`src/ilga_graph/run_log.py`:** `RunLogger` context manager and `append_startup_run()` write one JSONL record per run with task name, start/end, duration, status, and per-phase timings. `load_recent_runs(n, task=?)` for dashboards. Path from `ILGA_RUN_LOG` (default `.run_log.jsonl`).
  - **Instrumentation:** `scripts/ml_run.py` wraps main in `RunLogger("ml_run")` and logs each step (Backtest, Data Pipeline, Entity Resolution, Graph Embeddings, Bill Scoring, Coalitions, Anomaly Detection, Snapshot). `main.py` lifespan calls `append_startup_run(...)` after existing startup timing CSV. `scripts/scrape.py` wraps in `RunLogger("scrape")` with phases: Members+Bills, Votes+Slips, Analytics+Export (or Load cache + Export for export-only).
  - **Terminal:** `make logs` (or `make logs N=50`) runs `scripts/log_dashboard.py` — green/amber 2000s-hacker style, last N runs, per-run top phases, and a bottleneck summary (avg phase time per task).
  - **Web:** `GET /logs` renders `logs.html` — minimal 2000s-hacker UI (dark bg, monospace, green/amber), table of recent runs and a bottleneck table. Link from footer.
  - **Files:** `.gitignore` and `make clean` now include `.run_log.jsonl`.
  - **Next:** Optional: add run_log to other scripts (e.g. `ml_pipeline.py`, `scrape_votes.py`) for finer-grained history.

- **Phase 3: Node2Vec Graph Embeddings for Legislator Influence (2026-02-14):**
  - **Problem:** The current "Moneyball" heuristics treat legislators as independent entities — features like "Party" or "Sponsor Success Rate" miss the relational structure. If a Democrat frequently co-sponsors with three specific Republicans, a tabular model cannot detect this bipartisan bridge. The existing spectral embeddings in `coalitions.py` (32-dim Laplacian eigenvectors) are limited to linear structure.
  - **Solution — Node2Vec co-sponsorship embeddings:** Model the legislature as a weighted, undirected graph (nodes = legislators, edges = co-sponsorships, weights = frequency). Use Node2Vec (second-order random walks → Word2Vec skip-gram) to generate 64-dimensional dense vector embeddings for each legislator. These embeddings capture the structural position within the co-sponsorship network — legislators who frequently collaborate will have nearby vectors.
  - **New modules:**
    - `ml/graph_builder.py`: Builds the weighted co-sponsorship `nx.Graph` from `cache/bills.json`. Filters: only substantive bills (SB/HB), excludes bills with >40 sponsors (ceremonial fluff). All legislators from `dim_members.parquet` added as nodes.
    - `ml/node_embedder.py`: Node2Vec wrapper. Default hyperparameters: `dimensions=64, walk_length=30, num_walks=200, p=1.0, q=1.0, window=10`. Handles isolated nodes (zero vector). Saves to `processed/member_embeddings.parquet`. All params overridable via `ILGA_N2V_*` env vars.
  - **Feature integration** (`ml/features.py`):
    - New `build_embedding_features(df_bills)`: Loads pre-computed embeddings, maps each bill's `primary_sponsor_id` to its 64-dim vector. Produces columns `sponsor_emb_0` through `sponsor_emb_63`. Zero vector for unknown sponsors.
    - Wired into both `build_feature_matrix()` and `build_panel_feature_matrix()` — included in **both** full and forecast modes (embeddings are legislator-intrinsic, no temporal leakage).
    - Graceful degradation: if `member_embeddings.parquet` missing, returns zero-filled columns with a warning.
  - **Coalition upgrade** (`ml/coalitions.py`): `run_coalition_discovery()` now loads pre-computed Node2Vec embeddings for clustering instead of computing spectral embeddings. Falls back to spectral if Node2Vec file not available.
  - **Pipeline orchestration** (`scripts/ml_run.py`): New Step 3/7 "Graph Embeddings (Node2Vec)" inserted between Entity Resolution and Bill Scoring. Pipeline now has 8 steps (0–7).
  - **Makefile:** New `make ml-embed` target for standalone embedding generation.
  - **Dependencies:** Added `node2vec` to `pyproject.toml` `[ml]` extras (pulls in `gensim`).
  - **Tuning guidance:** `p` controls local vs global exploration — `p<1` favors homophily (local clustering), `q<1` favors structural equivalence. Default `p=1, q=1` is balanced (DeepWalk).
  - **Next steps:** Run `make ml-run`, compare model accuracy with/without embedding features, tune p/q parameters, consider aggregated co-sponsor embedding (mean of all co-sponsors per bill) as v2 enhancement.

- **Lint fixes before push (2026-02-14):**
  - Resolved 13 ruff errors: E501 line length (wrapped long lines in diagnostics_ml, main, features, rule_engine), F841 unused variables (coalitions.py `topic_results`, features.py `df_bills_sponsors`), E402 imports (moved `typing.Literal` and `action_classifier` to top of features.py; `action_classifier` to top of pipeline.py). `make lint` passes.
  - **E501 in main.py (bill detail last-action override):** Two more long lines at 3605/3611 — comment shortened; `days_since_action` expression broken out with `today = datetime.now().replace(...)` so line stays ≤100 chars. `make lint-fix` + `ruff format` applied; `make lint` passes.

- **Intelligence: Witness Slips tab (2026-02-13):**
  - New tab on `/intelligence/raw`: **Witness Slips** — demonstrates how organizations and lobbying groups influence bills via witness slip filings.
  - **Data:** `state.witness_slips_lookup` (per-bill slips), optional `state.ml.anomalies` for flagged bills. Route: `GET /intelligence/witness-slips`; partial: `_intelligence_witness_slips.html`.
  - **Content:** (1) Top organizations by total slip filings across all bills. (2) Bills by witness slip volume: bill, description, total / pro / opp / no pos, controversy %, flagged (suspicious coordination), top 5 orgs per bill with pro/opp breakdown. Filters: flagged only, search by bill/description/org.
  - **Organization name normalization:** `_canonical_organization_name()` maps duplicate/semantic variants to two canonical labels: **"No organization"** (NA, None, N/A, Not applicable, etc.) and **"Individual"** (Self, self, Myself, On behalf of self, Individual, Citizen, Family, Personal, Retired, etc.). Used when building top organizations (global and per-bill) so the Witness Slips tab shows consolidated counts.
  - **Future:** Slip–vote alignment (do members vote with/against slip majority?) can be added as a separate section or tab.

- **ML pipeline diagnostics (2026-02-13):**
  - Lightweight check of ML data in/out and model artifacts (no full `make ml-run`). Script: `scripts/diagnostics_ml.py` — run with `PYTHONPATH=src python scripts/diagnostics_ml.py`.
  - **Data inputs:** cache/bills.json (433 MB), members.json; processed dim_bills (11,722 rows), dim_members (180), fact_bill_actions (103,332), fact_witness_slips (394,203), vote facts present.
  - **ML outputs:** bill_scores.parquet (9,676 rows) with `forecast_score`, `forecast_confidence`, `prob_advance`, `prob_law`; Status, Law, and Forecast predictor PKLs load and expose `predict_proba`; model_quality.json (Status AUC 0.9966), forecast_quality.json present.
  - **API loader:** `load_ml_data()` returns `available=True`, 9,676 bill scores; `BillScore` has forecast fields. Feature imports (`build_feature_matrix`, `build_panel_feature_matrix`, `FeatureMode`, `FORECAST_DROP_COLUMNS`) OK.
  - **Tests:** All 277 tests pass (including test_panel_labels, test_moneyball). Full suite ~1.8s.
  - **Forecast audit + fix (see next bullet):** Audit found full-text leakage; fix applied, Forecast re-run (AUC now 0.945). ~~`forecast_quality.json` previously showed test ROC-AUC 1.0~~. ( “enrolled”/leakage — now fixed.)

- **Forecast model anti-leakage fix (2026-02-13):**
  - **Audit:** Forecast test AUC 1.0 was from label leakage: full-text TF-IDF included "enrolled"/"lrb104" from current bill text. Same bill could appear in train and test.
  - **Fix:** (1) Exclude full-text TF-IDF and full-text content metadata in Forecast mode (`FORECAST_DROP_COLUMNS` + skip in build_feature_matrix/build_panel_feature_matrix). (2) Train/test split by bill_id (70/30).
  - **Results:** Test ROC-AUC **0.945**; top features sponsor_count (67.7%), sponsor_bill_count, has_sponsor_id, tfidf_tax. `forecast_predictor.pkl` and `forecast_quality.json` updated. Run `make ml-run` to refresh `bill_scores.parquet`.

- **Phase 1: The "Truth" Upgrade — Anti-Leakage Forecast Model (2026-02-13):**
  - **Problem:** The current Status model achieves AUC 0.995, but `days_since_last_action` accounts for 58.9% of feature importance — it is essentially detecting dead bills, not predicting outcomes. A bill with no action in 180 days is definitionally "stuck" and the model learns that tautology. This provides zero utility for "this bill was just filed — will it pass?"
  - **Solution — Dual-model architecture:** Deployed a secondary **Forecast** model alongside the existing **Status** model:
    - **Status model** (unchanged): Uses all features including staleness, slips, action counts. Answers "Is this bill likely dead right now?"
    - **Forecast model** (new): Uses only intrinsic/Day-0 features — sponsor metrics (party, tenure, historical passage rate), bill text (TF-IDF), committee assignment, calendar features, and content metadata. Strictly excludes `days_since_last_action`, all action counts, witness slips, staleness flags, and rule-derived features. Answers "How likely is this bill to become law based on its content and sponsor?"
  - **Feature engineering refactor** (`ml/features.py`):
    - New `FeatureMode = Literal["full", "forecast"]` type and `FORECAST_DROP_COLUMNS` frozenset (31 columns excluded in forecast mode).
    - `build_feature_matrix(mode="full"|"forecast")` and `build_panel_feature_matrix(mode=...)` skip building slip/staleness/action/rule feature tables entirely in forecast mode.
    - `FORECAST_SNAPSHOT_DAYS = [0, 30]` — forecast panel trains at Day 0 (introduction) and Day 30 only.
  - **Forecast training pipeline** (`ml/bill_predictor.py`):
    - `run_forecast_model()` builds panel features with `mode="forecast"`, `snapshot_days=[0, 30]`, trains GradientBoosting (n=300, depth=6), calibrates with `CalibratedClassifierCV`, saves to `processed/forecast_predictor.pkl` + `processed/forecast_quality.json`.
    - Target: `target_law_after` (P(becomes law)) not `target_advanced_after`.
    - Scoring: builds single-row `mode="forecast"` matrix for all bills, produces `forecast_score` and `forecast_confidence` ("Low"/"Medium"/"High") per bill.
    - Integrated into `run_auto()` as Step 5c — runs automatically on every `make ml-run`.
  - **Data layer** (`ml_loader.py`): `BillScore` extended with `forecast_score: float` and `forecast_confidence: str`. Backward-compatible (defaults to 0.0/"" if columns missing).
  - **GraphQL API** (`main.py`): `BillPredictionType` extended with `forecast_score` and `forecast_confidence` fields. `_bill_score_to_type()` populates them. All template context dicts (intelligence summary, bill detail, member detail) include forecast data.
  - **UI** (templates): Predictions table has new "Forecast" column with score bar and confidence label (High/Medium/Low). Bill detail page shows Forecast P(Law) alongside Status P(Advance) and P(Law). Member detail bill table includes Forecast column. Intelligence summary bill-to-watch cards show forecast score. CSS classes `.forecast-conf`, `.forecast-high`, `.forecast-medium`, `.forecast-low` for visual distinction.
  - **Expected metrics:** Status model unchanged (AUC ~0.99). Forecast model target AUC 0.70–0.75 — this is expected and desirable. Top features should be `sponsor_hist_passage_rate`, `committee_id`, TF-IDF terms, not time-based. The Forecast model trades raw accuracy for genuine predictive power on active bills.
  - **Risk mitigation:** `CalibratedClassifierCV` ensures "30% forecast" means ~30% of such bills historically became law. Pipeline gracefully degrades if forecast model fails — defaults to 0.0/"".
  - **Next steps:** Run `make ml-run`, compare Forecast top features vs Status top features, verify no staleness/activity features in Forecast model. Consider adding "First 48h slips" as a future Forecast feature.

- **ILGA Rules Reference System — Bicameral v2 (2026-02-13):**
  - **Problem:** Action classification, pipeline stages, lifecycle logic, and feature engineering all used hardcoded strings and ad-hoc definitions. No grounding in the actual legislative rules that govern how bills move. The initial implementation only covered Senate rules; House rules (Rules Committee, Rule 19, different vote thresholds) were missing.
  - **Solution — Bicameral rule glossary + rule engine:** Extracted both the 104th GA Senate Rules (SR-4) and House Rules into a unified machine-readable glossary at `reference/ilga_rules.json` (v2.0.0). The glossary uses `senate_rule` / `house_rule` fields throughout instead of a single `rule` field. Key sections:
    - `stages`: 10 pipeline stages with bicameral rule citations (e.g. FILED → Senate Rule 5-1(d) / House Rule 37(d)).
    - `committees.procedural`: Senate Assignments (Rule 3-5/3-7/3-8) + House Rules Committee (Rule 15/18) + Executive + Executive Appointments.
    - `committees.senate_standing`: 29 Senate committees (Rule 3-4).
    - `committees.house_standing`: 46 House committees (Rule 11) — full enumeration.
    - `actions`: All 16 action categories with dual-chamber rule refs (e.g. committee_action → Senate Rule 3-11 / House Rule 22).
    - `outcomes`: Bicameral report types, tabling (Senate Rule 7-10 / House Rule 60), discharge (Senate 36 votes / House 60 votes).
    - `vote_thresholds`: Senate (59 members: 30/36/40) and House (118 members: 60/71/79) side-by-side.
    - `deadlines`: Senate (Rule 2-10/3-9) and House (Rule 9/19) schedules. Includes actual 2025 House deadlines.
  - **`ml/rule_engine.py` updated (bicameral):** Functions are now chamber-aware:
    - `get_stage_rule(stage, chamber=None)` — returns senate, house, or both rule citations.
    - `votes_required_for_passage(chamber)` — Senate 30, House 60.
    - `votes_required_for_override(chamber)` — Senate 36, House 71.
    - New: `votes_required_for_discharge(chamber)` — Senate 36, House 60.
    - New: `chamber_member_count(chamber)` — Senate 59, House 118.
    - Tooltip map updated with bicameral citations (e.g. "Override requires 3/5 (Senate 36, House 71)").
  - **`action_types.json` updated:** All 16 category `rule_reference` fields now cite both chambers (e.g. "Senate Rule 3-11; House Rule 22"). ~15 individual action patterns updated similarly.
  - **`bill_predictor.py` updated:** `rule_context` now uses chamber-aware vote counts — detects origin chamber from bill_id prefix (HB→house, SB→senate) for correct vote threshold display.
  - **Sources:** Senate Rules: `reference/104th_Senate_Rules.pdf`, House Rules: `reference/104th_House_Rules.pdf`.
  - **Prior implementation details** (still apply): `ClassifiedAction.rule_reference`, `build_rule_features()` (6 binary features), `BillScore.rule_context`, `ActionEntry.rule_reference`, template rule tooltips.

- **Full Bill Text Scraper + ML Feature Scaffold (2026-02-13):**
  - **Problem:** ML text features used only the short "Synopsis As Introduced" (typically 1-2 sentences) via TF-IDF. The model had almost no signal from the actual *content* of the bill — topics, length, complexity, citations to existing law.
  - **Solution — PDF-based full text scraping:** ILGA's FullText tab does NOT render large bills inline (shows "too large for display"). The only reliable method is downloading the PDF link always present on the FullText tab page and extracting text with `pdfplumber`.
  - **New scraper module** (`scrapers/full_text.py`): `_full_text_tab_url()` converts BillStatus URL to FullText tab URL (same pattern as witness_slips/votes). `_parse_pdf_link()` finds the "Open PDF" button. `_extract_text_from_pdf()` uses pdfplumber. `_clean_bill_text()` strips page headers, line numbers, normalises unicode. `scrape_bill_full_text()` orchestrates per-bill: fetch tab page → find PDF link → download (10MB size guard) → extract → clean. Returns cleaned text or `None` / `"[SKIPPED: PDF too large]"`.
  - **Resumable CLI** (`scripts/scrape_fulltext.py`): Modeled after `scrape_votes.py`. Progress file at `cache/fulltext_progress.json` tracks `scraped_bill_numbers`, `checked_no_pdf`, `skipped_too_large`. Batch saves every 25 bills (atomic write). SIGINT handler for clean Ctrl+C. Only SB/HB (skips resolutions). Flags: `--limit`, `--workers`, `--fast`, `--reset`, `--verify`, `--save-interval`. Real-time progress with word count and ETA.
  - **Model field** (`models.py`): Added `full_text: str = ""` to `Bill` dataclass.
  - **Serialization** (`scrapers/bills.py`): `_bill_from_dict()` reads `full_text`. `_bill_to_dict()` already includes it via `asdict()`. Full text preserved on re-scrape (carried forward like `vote_events`/`witness_slips`).
  - **Pipeline** (`ml/pipeline.py`): `process_bills()` now includes `full_text` in `dim_bills.parquet`.
  - **Length variance strategy** (bills range from <1 page to 500+ pages):
    - *Scraping:* Store full text with NO truncation. 10MB PDF size guard skips extreme outliers.
    - *Feature building:* Truncate to first 5000 words (~10-15 pages). `sublinear_tf=True` (log term frequency dampens long-bill dominance). `norm='l2'` (unit-length normalisation). Higher `min_df=5` / lower `max_df=0.7` vs synopsis.
    - *Length as explicit features:* `full_text_word_count`, `full_text_log_length` (log1p), `is_long_bill` (>10K words), `is_short_bill` (<500 words), `full_text_section_count`, `full_text_citation_count` (ILCS refs), `has_full_text` (binary).
  - **ML features** (`ml/features.py`): `build_full_text_features()` — separate TF-IDF on full text (1000 features, 5000-token truncation). `build_content_metadata_features()` — 7 numeric features. Both wired into `build_feature_matrix()` as third sparse block: `hstack([synopsis_tfidf, fulltext_tfidf, tabular])`. All return zeros gracefully when no full text exists — pipeline still runs without scraped text.
  - **Makefile:** `make scrape-fulltext` (default 100 bills), `make scrape-fulltext LIMIT=0` (all), `make scrape-fulltext FAST=1`.
  - **Speed optimisation v2 (2026-02-13):**
    - **Direct PDF URL prediction:** New `_predict_pdf_url()` in `scrapers/full_text.py` constructs the PDF URL directly from the bill number using the pattern `/legislation/{GA}/PDF/{GA}00{DocType}{Num}lv.pdf`, bypassing the FullText tab HTML fetch. Cuts HTTP requests from 2 per bill to 1 for most bills. Falls back to the original HTML flow on 404. Config constant `GA_NUMBER = GA_ID + 86` in `config.py`.
    - **Post-download delay removed:** The `time.sleep(request_delay)` after PDF download was eliminated (the function returns immediately; next request comes from a different thread).
    - **Refactored `scrape_bill_full_text()`:** New `bill_number` parameter enables the fast path. Extracted `_download_pdf()` (size guard + error handling) and `_extract_and_clean()` helpers to share between fast path and fallback.
    - **`--delay` CLI argument:** Exact delay control in `scripts/scrape_fulltext.py` (overrides `--fast` and default). `--delay 0` eliminates all inter-request delay. Makefile supports `DELAY=` variable.
    - **Save interval raised to 100:** Default save-every-N went from 25 to 100. The 150MB `bills.json` serialization was causing massive ETA spikes every 25 bills. Makefile supports `SAVE_INTERVAL=` variable.
    - **Expected improvement:** ~50-60% faster per bill (0.5-0.8s vs 1.2-1.8s), ETA from ~500 min to ~200-250 min with 15 workers.
    - **Usage:** `make scrape-fulltext LIMIT=0 WORKERS=15 DELAY=0` for maximum speed.
  - **Next steps:** Run `make scrape-fulltext LIMIT=0 WORKERS=15 DELAY=0` for all remaining SB/HB bills. Run `make ml-run` to evaluate model improvement. Tune `max_features`, `max_tokens`, TF-IDF params based on results.

- **Panel (Time-Slice) ML Dataset Expansion (2026-02-13):**
  - **Problem:** Training set was limited to ~one row per mature bill (120+ days old). Newer/"immature" bills contributed zero training signal, and even mature bills were only seen at their final state — never at intermediate stages.
  - **Solution — panel dataset (`ILGA_ML_PANEL=1`):** Instead of one row per bill, creates **multiple rows per bill** at snapshot dates (30, 60, 90 days after introduction). For each snapshot row: **features** use only data up to the snapshot date (no future leakage); **label** = "did this bill advance AFTER this snapshot?" Only included when we have observed long enough after the snapshot (90 days by default).
  - **New constants** (`features.py`): `SNAPSHOT_DAYS_AFTER_INTRO = [30, 60, 90]`, `OBSERVATION_DAYS_AFTER_SNAPSHOT = 90`.
  - **New function** `build_panel_labels(df_bills, df_actions)`: For each bill × snapshot day, splits actions into before/after the snapshot, assigns post-snapshot advance/law labels. Only creates rows where `snapshot_date + observation_days <= today`.
  - **Time-sliced feature builders**: `build_staleness_features`, `build_action_features`, `build_slip_features`, `build_sponsor_features`, and `build_committee_features` all accept optional `as_of_date` parameter. When set, they filter data to on-or-before that date and compute staleness/history relative to it instead of "now". TF-IDF and temporal features are snapshot-invariant (bill text and intro date don't change).
  - **New function** `build_panel_feature_matrix()`: Orchestrates the panel build — creates panel labels, iterates unique snapshot dates, builds features for each snapshot batch, concatenates into a single panel DataFrame. Returns the same tuple shape as `build_feature_matrix()` for drop-in compatibility with `run_auto()`.
  - **Integration** (`bill_predictor.py`): `run_auto(use_panel=None)` reads `ILGA_ML_PANEL` env var. When `"1"`, uses panel dataset for training/eval, then builds single-row features for scoring (so `bill_scores.parquet` always has one row per bill). Quality report records `panel_mode: true`.
  - **Activation:** Panel is the **default** for `make ml-run`. To revert to the legacy single-row training, set `ILGA_ML_PANEL=0 make ml-run`.
  - **Expected outcome:** 2-3x more training rows from the same bills (same bill at "in committee" at day 30, "passed committee" at day 60, etc.), capturing stage-transition patterns. Inference output unchanged — one row per bill.

- **Unified Panel + Full-Text Feature Paths (2026-02-13):**
  - **Problem:** `build_panel_feature_matrix()` was missing full-text TF-IDF and content-metadata features that `build_feature_matrix()` (used for scoring) included. This caused a feature-dimension mismatch at scoring time (sklearn `ValueError`), and meant the panel-trained model never learned from full-text signals.
  - **Fix (`features.py`):**
    - Added `build_full_text_features(df_bills_sub, max_features=1000)` and `build_content_metadata_features(df_bills_sub)` to `build_panel_feature_matrix()`, built once outside the snapshot loop (bill text is snapshot-invariant).
    - Joined `df_content_meta` into each snapshot's `df_feat_snap` so 7 content-metadata columns (`full_text_word_count`, `full_text_log_length`, `full_text_section_count`, `full_text_citation_count`, `has_full_text`, `is_long_bill`, `is_short_bill`) are present in panel training.
    - Updated panel `_to_sparse()` to hstack full-text TF-IDF alongside synopsis TF-IDF + tabular, matching the single-row path.
    - Updated panel `feature_names` to include `ft_tfidf_*` prefixed names.
    - Added `"full_text"` to `_NON_FEATURE_COLS` to exclude the raw text string from tabular features in both paths.
  - **Result:** Panel training and single-row scoring now produce identically-shaped feature matrices. No more dimension mismatch. Model learns from full-text signals when full text has been scraped. All 277 tests pass.

- **Committee Intelligence Tab + ML Committee Features (2026-02-13):**
  - **New Committees tab** (`/intelligence/committees`, `_intelligence_committees.html`): Added a "Committees" tab to the intelligence raw data dashboard showing every committee with sortable columns: name, chamber, total bills, advanced count, advancement rate (with visual bar), became law count, law rate, chair (linked to member detail), and member count. Client-side filters: All, Senate, House, Active (10+ bills), High Passage (>=30%). Sortable by any column. Shows visible/total count.
  - **Committee Insights section**: Below the table, three insight cards: Busiest Committees (top 10 by bill volume), Highest Passage Rates (top 10 among committees with 10+ bills), and Law Factories (top 10 by bills that became law). Each card shows the committee name, key stat, and detail line.
  - **Summary card**: Added "Committees" summary card to the intelligence/raw overview showing total committee count and number of active committees (10+ bills).
  - **New ML feature: `build_committee_features()`** (`features.py`): Extracts each bill's first substantive committee assignment from action history ("Assigned to"/"Referred to" actions, skipping procedural committees like Assignments, Rules, Executive). Computes 5 new features per bill:
    - `committee_advancement_rate`: Historical advancement rate of the bill's committee (only using bills introduced *before* this one — leakage-safe).
    - `committee_pass_rate`: Historical law rate of the committee (same leakage guard).
    - `committee_bill_volume`: How many bills the committee has handled historically.
    - `is_high_throughput_committee`: Binary flag for committees with >=30% advancement rate.
    - `has_committee_assignment`: Binary flag for whether the bill has a non-procedural committee assignment.
  - **Integrated into `build_feature_matrix()`**: Committee features are joined alongside sponsor, text, slip, temporal, action, and staleness features. Committee rates use -1 sentinel for missing data (consistent with other ratio columns). Binary flags added to the imputation lists.
  - **Tab button**: Added to `intelligence.html` between Influence and Model Quality tabs. Uses same HTMX pattern (`hx-get="/intelligence/committees"` → `#intel-content`).
  - **CSS** (`base.html`): Added `.committee-insights-grid`, `.committee-insight-card`, `.committee-insight-list`, `.committee-insight-stat`, `.committee-insight-detail` styles.
  - **Route** (`main.py`): New `GET /intelligence/committees` handler. Builds template-friendly dicts from `state.committees`, `state.committee_stats`, and `state.committee_rosters`. Computes chamber from code prefix, finds chair from roster, sorts by total bills. Generates three insight lists (top by volume, top by passage, law factories).
  - **Committee dashboard accuracy (2026-02-13):** Rules * Reports and Assignments * Reports showed 5k+ bills with 0 advanced — correct in our pipeline (we count "advanced" when last_action is Do Pass/Reported Out; bills in Rules have last_action "Referred to Rules") but misleading because those are **routing committees** (bills go there after passing a substantive committee). Changes: (1) **Procedural flag** (`main.py`): `_PROCEDURAL_COMMITTEE_NAMES` set; each committee dict gets `is_procedural`. (2) **Template** (`_intelligence_committees.html`): Procedural committees show "—" for advancement and law rate with "(routing)" tag and subtitle note. (3) **Insight cards**: Top by passage and Law Factories exclude procedural so they don't dominate. (4) **Percentage display**: Rates in (0, 1%) now show "<1%" instead of "0%" (fixes e.g. Assignments * Reports 5/3563 ≈ 0.14%). (5) **High Passage filter**: Excludes procedural rows. (6) **CSS** (`.committee-routing-tag` in `base.html`).

- **Data Quality Audit & Remediation (2026-02-13):**
  - **CRITICAL FIX: `total_action_count` data leakage removed** (`features.py`): The single most important feature (68.4% importance, CV AUC 0.999) was circular — bills that advance accumulate more actions by construction. Replaced with time-capped action counts (`action_count_30d`, `action_count_60d`, `action_count_90d`) and normalized velocity (`action_velocity_60d`). Expect model metrics to drop to realistic levels (80-90% AUC).
  - **Semantic imputation** (`features.py`): Replaced blanket `.fill_null(0)` with context-aware fills: counts → 0, ratios → -1 sentinel (distinct from "zero rate"), binary flags → 0. Added `has_no_slip_data` and `has_no_sponsor_history` indicator columns so the model can learn from missingness.
  - **Date normalization** (`normalize.py`, `scrapers/bills.py`, `scraper.py`): New `normalize_date()` converts all dates to ISO `YYYY-MM-DD` at scrape time and cache load time. Handles MM/DD/YYYY, "May 31, 2025", and "2025-05-31 17:00" formats. Applied to: `last_action_date`, action history dates, vote event dates, witness slip hearing dates.
  - **Chamber normalization** (`normalize.py`, `scrapers/bills.py`, `scraper.py`): New `normalize_chamber()` converts "Senate"/"House" to "S"/"H" consistently. Applied at scrape time and cache load time across all data models.
  - **Gold label expansion** (`ml/gold_labels.py`): Replaced the old 9-entry all-negative gold set with a proper 400-bill stratified sample (balanced ADVANCE/STUCK, proportional chamber/type). Auto-generated from action classifier labels with metadata (bill_number, sponsor, dates). Wired into `run_auto()` Step 8.
  - **Cache schema validation** (`normalize.py`, `scrapers/bills.py`, `scraper.py`): Added `validate_bill_dict()`, `validate_member_dict()`, `validate_bill_cache()` that check required fields (bill_number, leg_id, chamber) are present and non-null on every cache load. Warns on missing/empty recommended fields.
  - **Backtester label stability** (`ml/backtester.py`): New `LabelChurn` dataclass tracks how many bill labels changed between pipeline runs (STUCK→ADVANCE, ADVANCE→STUCK). Labels are now snapshotted alongside predictions. Backtest output reports churn rate, explaining the 75%-99% accuracy oscillation.
  - **Anomaly detection ground truth** (`processed/anomaly_labels_gold.json`, `ml/anomaly_detection.py`): Created 23 labeled examples (13 suspicious, 10 genuine). New `_tune_contamination_with_gold()` tests contamination 0.02-0.20 and picks the value maximizing F1 on gold labels. Replaces the hardcoded 8% contamination.

- **Influence Engine Fix + Low-Confidence UX + Leakage Documentation:**
  - **Influence engine bug fix** (`main.py`): `state.member_lookup` was keyed by member **name** but `compute_influence_scores()` looked up by **member_id** — every lookup failed, so `state.influence` was always `{}`. All influence scores were 0; graph, leaderboard, Power Movers, and coalition enrichment fell back to Moneyball. **Fix:** Added `state.member_lookup_by_id = {m.id: m for m in state.members}` and passed it into `compute_influence_scores()`. Also fixed Power Movers (executive summary) and intelligence member detail route to use the id-keyed lookup.
  - **Low-confidence indicator** (`_intelligence_predictions.html`, `intelligence.html`, `base.html`): Predictions with confidence <70% now show an amber left border, "Uncertain" label next to the confidence %, and a new "Low confidence (<70%)" filter checkbox. Accuracy history shows the 50–70% bucket is only ~56–76% accurate, so this visual cue prevents users from over-relying on uncertain predictions.
  - **Leakage documentation** (`features.py`): Added docstring note on `total_action_count` explaining it is outcome-correlated (~0.65 importance) but not strict leakage due to the time-based train/test split. Noted `build_action_features` as the conservative alternative (first 30 days only).

- **Jinja2 Template Syntax Fix (`intelligence_bill.html` line 64):**
  - **Bug:** `/intelligence/bill/:id` returned 500 Internal Server Error for every bill detail page.
  - **Root cause:** Line 64 used a Python-style ternary expression (`stages.index(s) < stages.index(bill.stage_label) if bill.stage_label in stages else false`) inside a `{% elif %}` block. Jinja2 doesn't support `X if condition else Y` ternary syntax in that position — it parsed the second `if` as a new statement and threw `TemplateSyntaxError: expected token 'end of statement block', got 'if'`.
  - **Fix:** Replaced with proper Jinja2 `and` operator: `{% elif bill.stage_label in stages and stages.index(s) < stages.index(bill.stage_label) %}`. Semantically identical — when `bill.stage_label` is not in `stages`, the short-circuit `and` prevents the `stages.index()` call.
  - **Verified:** `/intelligence/bill/155711` now returns 200 OK. All other intelligence endpoints unaffected.

- **Coalition Focus Area Bug Fix (`coalitions.py` — category matching overhaul):**
  - **Root cause:** All 10 coalitions were showing identical focus areas ("Education & State Government"). Three compounding bugs:
    1. **Missing space in action text:** ILGA actions use `"Assigned toTransportation"` (no space), but the regex expected `\s+` after "to", so it matched almost nothing.
    2. **Wrong-direction substring match:** The check `if cat.lower() in cn_lower` looked for the full category name in the committee name. Compound names like "Revenue & Pensions" never appeared as substrings of "Revenue", "Criminal Justice" never appeared in "Criminal Law", etc. Only short names like "Education" and "State Government" matched.
    3. **Senate-only committee codes:** `_CATEGORY_COMMITTEES` only had Senate committee codes (S-prefix). All House committees were unrepresented.
    4. **"Executive" catch-all:** The Senate/House "Executive" committee is a procedural catch-all (2,875 actions alone), not a policy indicator. Mapping it to "State Government" drowned out actual policy signals.
  - **Fix — `_extract_committee_name()`:** Rewrote regex to handle `\s*` (optional space) after "to". Added "Executive" and "Executive Appointments" to procedural filter list (alongside Rules, Assignments, etc.).
  - **Fix — `_CATEGORY_COMMITTEES`:** Expanded from 13 Senate-only codes to 70+ codes covering both Senate and House committees.
  - **Fix — `_CATEGORY_NAME_KEYWORDS`:** New keyword-based matching system. Each category has a list of substring keywords (e.g., Criminal Justice: "criminal", "judiciary", "public safety", "restorative justice", "gun violence"). Replaces broken full-name substring check.
  - **Fix — `_categorize_committee_name()`:** New function that uses keyword matching instead of direct category name comparison. Handles House committee names like "Judiciary - Criminal", "Revenue & Finance", "Labor & Commerce", etc.
  - **Result:** Category distribution now balanced — Revenue & Pensions (1,047), Criminal Justice (672), Healthcare (625), Education (478), State Government (379), Energy & Environment (268), Commerce (224), Labor (181), Insurance & Finance (176), Transportation (173), Agriculture (89), Housing (26). Coalition names are properly differentiated.
  - **Future enhancement:** Could use TF-IDF-like relative weighting (coalition category share vs. overall average) to further differentiate coalitions that vote on the same broad topics but with different emphasis.

- **Topic-Based Coalition Redesign (`coalitions.py` — complete rewrite):**
  - **Root cause of duplicate/useless coalitions:** THREE compounding issues:
    1. **Broken category filter:** Code filtered `action_category == "committee"` but the data uses `"assignment"` (from the action classifier update). Zero bills got categorized.
    2. **Silhouette score 0.115:** General (all-votes) clustering was finding noise, not structure.
    3. **Same categories for every bloc:** All blocs vote on the same bills (Criminal Justice/Healthcare dominate), so counting YES votes per category doesn't differentiate.
  - **New approach — per-topic voting coalitions (PRIMARY output):**
    - For each of 12 policy areas: find bills in that topic, find roll-call votes, compute per-member YES rate, segment into tiers: Champion (>=80%), Lean Support (65-80%), Swing (45-65%), Lean Oppose (25-45%), Oppose (<25%).
    - Results show clearly differentiated blocs per topic. E.g., Energy & Environment: 120 Champions (117D/3R), 30 Swing (0D/30R), 18 Lean Oppose (0D/18R). Labor: 125 Champions, 23 Lean Oppose, 4 Oppose. Housing: 122 Champions, 32 Lean Oppose, 9 Oppose.
    - Saved to `processed/topic_coalitions.json` — all member profiles per topic with YES/NO counts, rates, tier assignments.
  - **General clustering retained as SECONDARY:** Still runs (k=2..8), but honestly reports "Weak structure" when silhouette < 0.25. Now uses YES rate per category (not raw count) for focus areas.
  - **Category mapping fix:** New `build_bill_categories()` uses `action_category == "assignment"` (correct filter). Maps 3,110 bills. 12 topics with enough data.

- **Legislative Power Map (`/explore`) — Interactive Graph Visualization:**
  - **New route: `GET /explore`** — full-viewport D3.js v7 force-directed graph of legislators.
  - **New API: `GET /api/graph?topic=X&zip=Y&focus=relevant|all`** — JSON endpoint returning nodes, edges, your_legislators, topic_committees, meta. **Focus:** `focus=relevant` (default) — when a **topic is selected**, returns only members on that topic's committees + your legislators; when **no topic**, returns top 50 by influence + your legislators. `focus=all` returns all 180. Topic filter (12 categories) and ZIP lookup supported.
  - **`moneyball.py`:** Renamed `_build_cosponsor_edges()` to `build_cosponsor_edges()` (public). Now called during startup and stored on `AppState.cosponsor_adjacency` for the graph API.
  - **`main.py` (AppState):** Added `cosponsor_adjacency: dict[str, set[str]]` field. Step 2a in lifespan computes adjacency from member co-sponsorship data (180 nodes, ~15k raw edges).
  - **`main.py` (graph API):** Smart edge pruning: important members (topic-relevant, user's legislators, top 20 by influence) keep all edges; others capped at 8 edges per node. When focus=relevant, edges filtered to only those between included nodes.
  - **`templates/explore.html`:** Full D3 visualization with:
    - **Force layout:** `forceLink` (co-sponsorship), `forceCharge` (repulsion), `forceCollide` (overlap prevention), `forceY` (influence hierarchy — high influence at top), `forceX` (gentle party clustering — D left, R right).
    - **Node appearance:** Circle size = influence score with **power scale** (exponent 1.8, range 2–34px) so high-influence members are visually dominant; color = party (blue D, red R, gray other); stroke = gold ring for user's legislators, blue ring for topic-relevant members.
    - **Show control:** "Relevant only" (default) vs "All 180 legislators" dropdown; status bar shows "(relevant only)" when applicable.
    - **Interactions:** Zoom/pan (d3.zoom), drag nodes (d3.drag), hover tooltip (name, role, influence, laws passed), click detail panel.
    - **Topic filter:** Dropdown of 12 policy categories. Re-queries API, dims non-relevant members to 12% opacity, preserves committee connections.
    - **ZIP input:** Highlights user's senator and representative with gold border + always-visible labels.
    - **Detail panel:** Right-side panel on click showing influence score bar, key stats (laws passed, cross-party rate, Moneyball), committee memberships with leadership roles, influence signals.
    - **Legend:** Party colors, size scale, "your legislator" indicator.
    - **Labels:** Top 15 by influence + user's legislators + topic committee chairs always labeled; others on hover.
  - **Navigation:** `/explore` linked from advocacy index, intelligence dashboard, and base footer. Cross-links between all three views.
  - **Auth:** `/explore` and `/api/graph` exempt from API key middleware (public-facing).
  - **Bug fix (committee filter):** When a topic/committee is selected with "Relevant only", the graph now returns only members on that topic's committees + your legislators. Previously it also included the global top 50 by influence, so unrelated nodes appeared. (`main.py` graph API: when `focus=relevant` and `topic_member_ids` non-empty, relevant set is topic_member_ids | your_legislator_ids only.)

- **Predicted Destination & Law Model (v6 — "advance to where?"):**
  - **Problem:** The Prediction column showed only "ADVANCE" or "STUCK" — too vague. Bills that advance past committee vs. become law are very different outcomes for advocacy.
  - **Second ML model:** Trained a separate GradientBoosting model on `target_law` labels (P(becomes law)) alongside the existing advance model. Same feature matrix, different target. Law model evaluated with its own ROC-AUC on the held-out test set.
  - **Predicted destination logic (`_compute_predicted_destination()`):** Combines P(advance), P(law), lifecycle status, and current stage into a human-readable destination:
    - **→ Law** — P(law) >= 0.5, or bill already at PASSED_BOTH/GOVERNOR with P(law) >= 0.3
    - **→ Passed** — Already at FLOOR_VOTE/CROSSED_CHAMBERS with P(advance) >= 0.5
    - **→ Floor** — P(advance) >= 0.5 but P(law) < 0.5
    - **Stuck** — P(advance) < 0.5
    - **Became Law** / **Vetoed** — Terminal bills show actual outcome
  - **Confidence cap:** Capped at 99% (`CONFIDENCE_CAP = 0.99`) to avoid false certainty from rounding.
  - **New parquet columns:** `prob_law`, `predicted_destination` added to `bill_scores.parquet`.
  - **Data model:** `BillScore` in `ml_loader.py` gains `prob_law: float` and `predicted_destination: str`.
  - **GraphQL:** `BillPredictionType` now exposes `prob_law` and `predicted_destination`.
  - **Prediction table (`_intelligence_predictions.html`):** New "P(Law)" column with score bar. "Prediction" column replaced with "Predicted Dest." showing color-coded destination badges (green → Law, yellow → Floor, blue → Passed, red Stuck, dark green Became Law, gray Vetoed). Filters updated: "Destination" group (→ Law, → Floor, Stuck) replaces old "ADVANCE only"/"STUCK only" checkboxes.
  - **Bill detail page (`intelligence_bill.html`):** Prediction card shows destination badge + both probability bars (P(Advance) and P(Becomes Law)) side by side.
  - **Member detail page (`intelligence_member.html`):** Top bills table gains "Predicted Dest." column with destination badges.
  - **Executive summary (`intelligence_summary.html`):** Bills to Watch cards show destination badges instead of ADVANCE/STUCK tags.
  - **Raw data overview (`intelligence.html`):** Summary card shows destination-based counts (N → Law, N → Floor, N Stuck) instead of advance/stuck counts.
  - **CSS (`base.html`):** 7 destination badge styles: `.dest-law` (green), `.dest-floor` (amber), `.dest-passed` (blue), `.dest-governor` (teal), `.dest-stuck` (red), `.dest-became-law` (solid green), `.dest-vetoed` (gray).
  - **Saved models:** `processed/bill_predictor.pkl` (advance) + `processed/bill_law_predictor.pkl` (law).

- **Structured Action Classification System (`action_types.json`, `action_classifier.py`):**
  - **New reference file:** `src/ilga_graph/ml/action_types.json` — comprehensive reference of every IL legislative action type, organized into 16 categories: Introduction & Filing, Committee Assignment, Committee Action, Deadlines & Re-referrals, Floor Process, Floor Vote, Cross-Chamber, Concurrence, Governor Action, Veto Override Process, Enacted Into Law, Co-Sponsorship, Amendment Actions, Resolution Process, Executive Appointments, Procedural. Each action has a pattern, match type, human-readable meaning, and outcome signal (positive_terminal, positive, positive_weak, neutral, negative_weak, negative, negative_terminal).
  - **New classifier module:** `src/ilga_graph/ml/action_classifier.py` — classifies raw ILGA action text into structured `ClassifiedAction` objects with: `category_id`, `category_label`, `outcome_signal`, `meaning`, `is_bill_action` (vs amendment), `progress_stage`. Handles ILGA's inconsistent formatting (missing spaces, vote tallies, amendment prefixes). 99.3% coverage across 103,332 actions.
  - **Pipeline update:** `pipeline.py` now imports `action_category_for_etl` from the classifier instead of the old naive `_classify_action()` with substring matching. New categories in parquet: introduction (20.5%), cosponsor (19.2%), assignment (18.5%), floor_process (11.5%), deadline (7.7%), amendment (7.6%), committee_action (4.5%), cross_chamber (2.5%), floor_vote (2.1%), enacted (1.2%), governor (0.9%), concurrence (0.9%), appointment (0.8%), procedural (0.7%), resolution (0.6%), veto_process (0.01%).
  - **Features refactored:** `features.py` now delegates all action classification to the `action_classifier` module: `_bill_has_negative_terminal()` checks for `negative_terminal` signals, `_bill_advanced()` checks for positive signals in key categories, `_bill_lifecycle_status()` uses `bill_outcome_from_actions()`, `compute_bill_stage()` uses classifier's `progress_stage` mapping. All old token lists (`_ADVANCED_TOKENS`, `_SIGNED_TOKENS`, `_AMENDMENT_PREFIXES`, `_STAGE_DEFINITIONS`) replaced by the classifier.
  - **Data model:** `ActionEntry` in `models.py` gains 4 new fields: `action_category`, `action_category_label`, `outcome_signal`, `meaning`. Populated at cache-load time in both `scrapers/bills.py` and `scraper.py`.
  - **GraphQL API:** `ActionEntryType` now exposes `action_category`, `action_category_label`, `outcome_signal`, `meaning`. New `actionTypesReference` query returns the full action types reference (categories with action patterns, meanings, signals) for any frontend to consume.
  - **Bill detail page:** `intelligence_bill.html` now shows a full action history timeline with color-coded dots (green=positive, red=negative, gray=neutral), category badges, outcome signal badges, and human-readable meaning for each action. Replaces the old single "Last Action" line.
  - **CSS:** Extensive timeline styling in `base.html` — `.action-timeline-*`, 16 category-specific badge colors (`.action-cat-*`), outcome signal badges (`.signal-*`), timeline dot colors by signal.

- **ML Pipeline Bug Audit (Phase A) + Intelligence Story Redesign (Phase B):**
  - **Bug fix: TF-IDF zero-vector fallback** (`features.py`): `_to_sparse()` was using `tfidf_id_to_idx.get(bid, 0)` which silently reused bill #0's TF-IDF features for any missing bill. Now uses a pre-computed zero vector so missing bills get no text signal instead of wrong text signal.
  - **Bug fix: KeyError guards** (`features.py`): `member["party"]` and `member["sponsored_bill_count"]` replaced with `.get()` to handle incomplete member dicts.
  - **Bug fix: Dynamic majority party** (`features.py`): `is_majority = is_democrat` was hardcoded for IL 104th GA. Now computed dynamically from member data party counts.
  - **Bug fix: Sponsor rate null guard** (`features.py`): `sponsor_filed` counter was incrementing for bills with unknown `target_advanced` labels (from left join), inflating the denominator. Now only counts bills with known labels.
  - **Bug fix: Vote tie handling** (`pipeline.py`): `"passed" if total_yea > total_nay else "lost"` classified ties as "lost". Now handles three outcomes: "passed", "lost", "tied".
  - **Bug fix: Anomaly NaN guard** (`anomaly_detection.py`): Added `fill_nan(0).fill_null(0)` before `to_numpy()` and explicit empty-matrix guard.
  - **Defensive logging** (`pipeline.py`, `bill_predictor.py`, `features.py`): Silent date-parsing failures now log warnings/debug messages with the unparseable value and bill ID, tracked with deduplication to avoid log spam.
  - **Intelligence Story Redesign:**
    - **Executive summary** (`intelligence_summary.html`, `/intelligence`): New narrative-driven landing page replacing the old tabbed data dump. Sections: Model Confidence banner (trust level, ROC-AUC, accuracy %), Bills to Watch (open bills with interesting ML signals — high confidence, surprises, and warnings), Power Movers (top 8 influencers as clickable cards), Coalition Landscape (voting blocs with party bars, cohesion, focus areas, top influencer), Anomaly Alert (top 5 flagged bills).
    - **Member deep-dive** (`intelligence_member.html`, `/intelligence/member/:id`): Full influence profile for one legislator. Sections: influence score bar + rank, narrative summary (auto-generated prose), influence component breakdown (4 cards: Moneyball, Bridge, Swing Votes, Sponsor Pull with contextual descriptions), key signals list, Moneyball effectiveness stats (laws passed, effectiveness rate, magnet score, bridge score, collaborators), notable bills table (top 10 by P(advance)), coalition membership.
    - **Bill deep-dive** (`intelligence_bill.html`, `/intelligence/bill/:id`): Full prediction context for one bill. Sections: prediction card (shows actual outcome for terminal bills, model prediction for open bills), visual pipeline progress (7-stage stepper with current stage highlighted), sponsor context card (linked to member profile, influence badge, sponsor lift), public engagement (witness slip counts, anomaly alert if flagged), last action.
    - **Raw data tables** (`intelligence.html` moved to `/intelligence/raw`): Original tabbed interface preserved for power users/debugging. Header updated with back-link to narrative summary.
    - **CSS** (`base.html`): Full styling for story pages — confidence banner, story card grid, bill/member profile layouts, pipeline stepper visualization, coalition bars, anomaly rows, component cards, signal lists.
    - **Navigation**: `/intelligence` → executive summary, `/intelligence/raw` → data tables, `/intelligence/member/:id` → member profile, `/intelligence/bill/:id` → bill profile. Cross-links between all views (bills link to sponsors, sponsors link to bills, summary cards link to detail pages).

- **Prediction Table v5 — Critical Fixes (`features.py`, `bill_predictor.py`, `ml_loader.py`, template):**
  - **Bug fix: Vetoed bills labeled ADVANCE** (`features.py`): `_bill_advanced()` was returning `True` for vetoed bills because they had "Passed Both Houses" actions before the veto. Added `_bill_has_negative_terminal()` check — any bill with a veto, tabling, or sine die action is now classified as NOT advanced. Added `_NEGATIVE_TERMINAL_TOKENS` list (Vetoed, Total Veto, Amendatory Veto, Item Veto, Tabled, Motion to Table - Lost, Rule 19, Session Sine Die).
  - **Bug fix: Prediction override for terminal bills** (`bill_predictor.py`): In `score_all_bills()`, after the model scores each bill, we now check the bill's lifecycle status. Passed/signed bills are forced to ADVANCE with 100% confidence. Vetoed/dead bills are forced to STUCK with 0% prob_advance and 100% confidence. Only OPEN bills keep the model's prediction. This prevents absurd situations like vetoed bills showing "ADVANCE 98%".
  - **Lifecycle semantics fix (OPEN | PASSED | VETOED only)** (`features.py`, `bill_predictor.py`, `ml_loader.py`, templates): Lifecycle is now **OPEN | PASSED | VETOED** only. **VETOED** means the governor actually vetoed the bill (confirmed terminal). Everything else — including tabled, postponed, Rule 19 re-referrals, sine die references — is **OPEN**. Status "DEAD" in `classify_stuck_status()` only triggers on actual governor vetoes (via `_bill_has_negative_terminal()`), not procedural actions. This fixes bugs like HB2432 showing Lifecycle/Status VETOED while Pipeline Stage was "In Committee".
  - **CRITICAL FIX: False-positive token matching** (`features.py`): The previous substring matching (`"Rule 19" in action_text.lower()`) produced **thousands of false positives**:
    - **"Rule 19"** matched 3,116 actions — ALL were procedural re-referrals (`Rule 19(a) / Re-referred to Rules Committee`), zero actual deaths.
    - **"Tabled"** matched 215 actions — almost all were amendments being tabled (`House Floor Amendment No. 1 Tabled`), not bills.
    - **"Postponed"** matched 265 actions — all routine committee postponements (`Postponed -Transportation`).
    - **"sine die"** / **"Session Sine Die"** matched 199 actions — all references to the previous 103rd GA session (`Due to Sine Die of the 103rd General Assembly`), not the current bill dying.
    - Only **6 actual veto actions** exist across all 11,722 bills (2 "Governor Vetoed" + 4 "Total Veto" variants).
    - **Fix:** New `_is_bill_action()` function filters out amendment/sub-item actions (prefixes like "House Floor Amendment", "Senate Committee Amendment", etc.). New precise veto matching: only "governor vetoed", "total veto stands", "amendatory veto" (not override/motion), "item veto" (not override/motion). Removed `_NEGATIVE_TERMINAL_TOKENS` tuple, `_VETO_TOKENS` list, and `_SESSION_END_TOKENS` list — all replaced by precise matching logic. `classify_stuck_status()` now delegates to `_bill_has_negative_terminal()` for DEAD classification instead of maintaining its own token list.
    - **Result:** 10,974 OPEN (93.6%), 745 PASSED (6.4%), 3 VETOED (0.03%) — correct distribution. Previously thousands of bills were misclassified as dead/vetoed.
  - **New: Bill lifecycle status** (`features.py`): `_bill_lifecycle_status()` classifies into three states: PASSED (signed/public act), DEAD (vetoed only), OPEN (everything else, including tabled/sine die). Added `lifecycle_status` column to `bill_scores.parquet`, `BillScore` dataclass in `ml_loader.py`.
  - **New: Staleness features** (`features.py`): Added `build_staleness_features()` producing 7 new features: `days_since_last_action`, `days_since_intro`, `action_velocity` (actions/day), `is_stale_90`, `is_stale_180`, `total_action_count`, `has_negative_terminal`. These let the model learn that idle bills (200+ days) are unlikely to advance. Integrated into `build_feature_matrix()`.
  - **Template: Lifecycle column + filters** (`_intelligence_predictions.html`): New "Lifecycle" column with color-coded badges (OPEN=blue, PASSED=green, VETOED=red, DEAD=gray). Closed bills get muted row opacity. New lifecycle filter checkboxes (Open, Passed, Vetoed, Dead). For terminal bills, P(Advance) shows "Passed" or "Failed" instead of a percentage, Prediction shows actual outcome (PASSED/VETOED/DEAD) not model output, and Confidence shows "—" instead of a percentage.
  - **Filter bar + grouped checkboxes** (`_intelligence_predictions.html`, `intelligence.html`, `base.html`): Search bar filters by bill number, description, and sponsor (name) as you type. Quick filters kept as checkboxes but grouped into three sections: **Prediction** (Forecasts only, ADVANCE only, STUCK only, High confidence ≥80%), **Lifecycle** (Open, Passed, Vetoed, Dead), **Status** (Stagnant, Slow, New). CSS: `.intel-filter-bar`, `.intel-filter-search`, `.intel-filters-grouped`, `.intel-filter-group`, `.intel-filter-group-label`. Search and filter logic live in the parent page (`intelligence.html`) and the search input is bound on `htmx:afterSettle` so filtering works when the Predictions tab is loaded via HTMX (scripts in swapped content are not executed by default).
  - **Sortable table** (`_intelligence_predictions.html`): All column headers are now clickable to sort. Supports both string and numeric sorting with ascending/descending toggle. Arrow indicators show current sort direction.
  - **CSS** (`base.html`): Added `.lifecycle-badge` with `.lifecycle-open/passed/vetoed/dead` color variants, `.closed-bill` muted row opacity, `.intel-outcome-vetoed` badge style, `.confidence-actual` for dashed placeholder on terminal bills, `.sortable` header hover/cursor styles, `.sort-arrow` indicator.

- **True Influence Engine (`influence.py` + Moneyball upgrades):**
  - **Betweenness centrality** (`moneyball.py`): Added `betweenness_centrality()` function using networkx. Computes normalized betweenness for every member in the co-sponsorship graph — measures how often a legislator lies on shortest paths between others (bridge/connector influence). Added `betweenness` field to `MoneyballProfile`. New "Bridge Connector" badge for betweenness ≥ 0.02. Updated `analytics_cache.py` serialization, `schema.py` GraphQL type, and `exporter.py` Obsidian rendering.
  - **Vote pivotality** (`influence.py`): `compute_vote_pivotality()` — for each close roll call (margin ≤ 5), scores members who voted with the winning side as "pivotal." Tracks `close_votes_total`, `pivotal_winning`, `pivotal_rate`, `swing_votes` (margin-of-1 true tiebreakers), `floor_pivotal` vs `committee_pivotal`. Returns `MemberPivotality` per legislator.
  - **Sponsor pull** (`influence.py`): `compute_sponsor_pull()` — uses ML bill scores (`prob_advance`) to measure whether a member's sponsorship/co-sponsorship is associated with higher success. Computes `sponsor_lift` (primary sponsor avg vs chamber avg) and `cosponsor_lift` (co-sponsor avg vs chamber avg). Blended `pull_score` weights primary 2× since it's a stronger signal.
  - **Unified InfluenceScore** (`influence.py`): `compute_influence_scores()` — blends four signals into a single 0–100 composite:
    - **Moneyball** (40%): outcome effectiveness, network, institutional power.
    - **Betweenness** (20%): structural bridge influence between blocs.
    - **Pivotality** (20%): swing vote power on close calls.
    - **Sponsor Pull** (20%): ML-predicted bill success association.
    - Produces `InfluenceProfile` with rank_overall, rank_chamber, influence_label (High/Moderate/Low), and human-readable `influence_signals` (e.g. "Bridges legislative blocs", "Cast deciding vote 3x").
  - **Coalition influence enrichment** (`influence.py`): `enrich_coalitions_with_influence()` — for each voting bloc, identifies top influencer (highest influence score) and bridge member (highest betweenness). Reports avg influence, max influence, and high-influence member count per bloc.
  - **App startup Step 8** (`main.py`): After ML load, computes pivotality (8a), sponsor pull (8b), influence scores (8c), coalition influence (8d). All stored on `AppState`.
  - **Intelligence dashboard** (`/intelligence/influence`): New "Influence" tab with:
    - Leaderboard table: rank, name, chamber, party, influence score bar, label, component breakdowns (Moneyball%, Betweenness%, Pivotality%, Pull%), human-readable signals.
    - Client-side filters: All, High Only, Senate, House.
    - Coalition Influence section: cards per bloc showing top influencer, bridge member, avg influence.
  - **GraphQL**: New `influence_leaderboard(chamber, limit)` query returns `InfluenceProfileType` list. `InfluenceProfileType` type with all component scores, pivotality details, sponsor lift. Added `influence` field to `MemberType.from_model()`.
  - **Advocacy cards**: `_build_influence_dict()` helper adds influence data to each card dict (score, label, signals, component breakdowns, pivotality details, sponsor lift).
  - **CSS** (`base.html`): Influence score bars (high=green, moderate=gold, low=red), `.influence-signal-tag`, `.intel-coalition-grid`, `.intel-coalition-card`, `.intel-trust-high` / `.intel-trust-low` label styles.



- **Power Card Redesign (advocacy card layout overhaul):**
  - **Card layout restructured** to match advocacy professional mockup. Old layout had data scattered across separate expandable sections. New layout consolidates the key "power signals" into a single up-front narrative box.
  - **New card structure (top to bottom):**
    1. **Badge row:** Role label (Your Senator / Power Broker / etc.) + power badges (LEADERSHIP, COMMITTEE CHAIR, TOP 5% INFLUENCE) side by side.
    2. **Name line:** `"Senator Jane Smith (R-Senate District 47)"` — inline party abbreviation and district.
    3. **"Why this person matters" box** (blue-bordered callout) with bulleted power signals:
       - **Committee chair positions** — "CHAIR: Transportation — Decides which bills get hearings — Advanced 29 of 51 bills (57% success rate) - 29 became law"
       - **Ranking** — "#32 of 60 senators" with Top N% highlight when in top 10%
       - **Network** — "Co-sponsors with 174 legislators — 56 Republicans, 118 Democrats — high bipartisan reach" + co-sponsor pull multiplier + co-sponsor passage multiplier
       - **Track record** — "1108 YES / 0 NO across 1144 votes — Votes with party 99.1% — breaks ranks 8x (potentially persuadable)"
       - **Workload** — "Currently managing N active bills"
    4. **Compact scorecard** (always visible, not collapsible): Laws passed with caucus comparison ("2.0x caucus average"), cross-party %, co-sponsor pull with chamber average, Moneyball with rank.
    5. **Contact row:** Phone / Email / ILGA Profile — all inline with icons.
    6. **Why this target + Script Hint** — existing evidence-based content.
    7. **Expandable details:** Voting Record, Committee Assignments, Full Legislative Scorecard (moved to bottom, still collapsible).
  - **`_member_to_card()` (main.py):** Added `rank_chamber`, `chamber_size`, `rank_percentile`, `party_abbr`, `email`, `role`, `active_bills` to card dict. Active bill count computed by checking `last_action` for non-terminal statuses.
  - **`moneyball.py`:** Added `passage_rate_vs_caucus` and `caucus_avg_passage_rate` fields to `MoneyballProfile`. New computation in `compute_moneyball()` calculates party+chamber average passage rate (caucus avg) and each member's multiplier against it. E.g., "2.0x caucus average" means this member passes bills at twice the rate of their party peers.
  - **`analytics_cache.py`:** Updated serialization to include new MoneyballProfile fields.
  - **`_results_partial.html`:** Complete rewrite of the `member_card` macro. Influence network section and old `<dl>` stats section removed from inline — data now consolidated into the power context box and compact scorecard.
  - **`base.html`:** Added CSS for `.card-badge-row` (flex row), `.card-name-meta` (inline party/district), `.power-context` (blue-bordered box with `.power-context-title`, `.power-context-list`, `.power-signal`, `.power-sub`, `.power-highlight`), `.compact-scorecard` (beige bordered box), `.compact-stats`, `.caucus-compare` (green bold for multipliers), `.contact-row` (flex row with icon prefixes for phone/email/profile).

- **Voting Pattern History (Track Record) on advocacy cards:**
  - **`voting_record.py` (new module):** Created `MemberVoteRecord` dataclass (bill_number, bill_description, date, vote=YES/NO/PRESENT/NV, bill_status, vote_type) and `VotingSummary` dataclass (total_floor_votes, yes/no/present/nv counts, yes_rate_pct, party_alignment_pct, party_defection_count, records list). Core functions: `build_member_vote_index()` — iterates all vote events and builds a reverse index (member_name -> VotingSummary) with per-member vote records sorted by date descending; `_compute_all_party_alignment()` — single-pass computation across all floor votes determining each party's majority direction per event then scoring each member's alignment (ties are skipped; PRESENT/NV counted as defections); `build_all_category_bill_sets()` — precomputes bill_number sets for each policy category from committee-bill mappings; `filter_summary_by_category()` — returns a new VotingSummary filtered to only category-relevant bills with recalculated counts.
  - **`main.py` (AppState + lifespan):** Added `state.member_vote_records` (dict[str, VotingSummary]) and `state.category_bill_sets` (dict[str, set[str]]) fields. New Step 4b in lifespan computes both after vote normalization. `_member_to_card()` now accepts optional `category` parameter; when provided, filters voting record to category bills. Returns `voting_record` dict with: records (up to 10 most recent), yes/no/present/nv counts, yes_rate_pct, party_alignment_pct, party_defection_count, is_persuadable flag, and category_label. All `_member_to_card()` calls in `advocacy_search()` pass the selected category.
  - **`_results_partial.html`:** New collapsible "Voting Record" `<details>` section between Legislative Network and Committee Assignments. Shows category-qualified title when filtered (e.g., "Voting Record: Transportation Bills (5 floor votes)"). Each vote row displays: bill number (bold), truncated description, color-coded vote indicator (YES=green, NO=red, PRESENT=amber, NV=gray), bill outcome. Summary bar shows YES/NO pattern with yes rate. Party alignment line when data available. "Potentially persuadable" callout (amber left-border box) when party_defection_count > 0 — the key advocacy signal.
  - **`base.html`:** Added CSS for `.card-voting-record` (collapsible, matching existing pattern), `.vote-list`, `.vote-row` (flexbox layout), `.vote-bill`, `.vote-desc`, `.vote-indicator` with `.vote-yes`/`.vote-no`/`.vote-present`/`.vote-nv` colour variants, `.vote-outcome` with status-based colours, `.vote-summary`, `.vote-pattern`, `.persuadable-callout` (amber left-border on cream background).
  - **Why it matters:** Tells advocacy groups: (1) Is this person persuadable? (votes against party sometimes = yes), (2) Do they support our issue? (voted YES on X of Y related bills). Section only renders when vote data exists, so no empty sections.

- **Institutional Power Badges (visual hierarchy):**
  - **`moneyball.py`:** Added `PowerBadge` dataclass (`label`, `icon`, `explanation`, `css_class`) and `compute_power_badges()` function. Three additive badge types: (1) **LEADERSHIP** — awarded when `institutional_weight >= 0.25`; explanation dynamically built from `member.role` with tier-specific context (top chamber leader / committee leader / party management). (2) **COMMITTEE CHAIR** — awarded for each committee where member is Chair (not Vice-Chair); explanation names the committee(s) and explains gatekeeper power. Multiple chairs consolidated into one badge. (3) **TOP N% INFLUENCE** — awarded when `rank_chamber` is in top 5% of their chamber (`ceil(chamber_size * 0.05)`); explanation shows exact rank and chamber size (e.g., "Ranked #3 of 59 senators").
  - **`main.py` (_member_to_card):** Computes chamber size from `state.moneyball.rankings_house` / `rankings_senate` lengths. Calls `compute_power_badges(mb, committee_roles, chamber_size)` and serialises results as `power_badges` list of dicts in each card. Imported `compute_power_badges` from moneyball module.
  - **`_results_partial.html`:** New `power-badges` div at the top of each card macro, rendered *above* existing role/achievement badges. Each badge uses native `<details>/<summary>` for click-to-expand explanations — no JavaScript required. Icons via CSS `::before` pseudo-elements with Unicode text presentation selectors.
  - **`base.html`:** Added CSS for `.power-badges` (flex row), `.power-badge-wrapper` (inline details), `.power-badge` (summary styled as badge with marker removal for all browsers), three colour variants (`.power-badge-leadership` navy, `.power-badge-chair` gold-brown, `.power-badge-influence` red), icon prefix classes (`.power-icon-leadership/chair/influence` using Unicode shield/lightning/fire), `.power-badge-explanation` (tooltip-like box below badge on expand). Hover state with brightness filter for clickability affordance.

- **Influence Network Visualization (human-readable power picture):**
  - **`moneyball.py` (MoneyballProfile + compute_moneyball):** Added 7 new fields to `MoneyballProfile`: `collaborator_republicans`, `collaborator_democrats`, `collaborator_other` (party breakdown of co-sponsorship peers), `magnet_vs_chamber` (member magnet / chamber avg magnet multiplier, e.g. "1.2x"), `cosponsor_passage_rate` (passage rate of bills this member co-sponsors), `cosponsor_passage_multiplier` (vs **chamber median co-sponsor rate**, not raw chamber passage rate -- avoids selection bias where popular/likely-to-pass bills naturally attract more co-sponsors), `chamber_median_cosponsor_rate` (the baseline, shown in parenthetical for transparency). New Step 3b in `compute_moneyball()` computes all three metrics: iterates adjacency peers for party breakdown, computes per-chamber magnet averages, and calculates co-sponsored bill passage rates vs peer-normalised baselines.
  - **`analytics_cache.py`:** Updated `_profile_to_dict()` to serialize new fields. Deserialization (`MoneyballProfile(**d)`) handles old caches gracefully via defaults.
  - **`main.py` (_member_to_card):** Added `influence_network` dict to card data with all computed metrics. Bipartisan label computed from party balance of network peers but NOT displayed on the card (removed to avoid confusion with the cross-party support metric which measures their own bills). Included in return dict for template access.
  - **`_results_partial.html`:** New "Legislative Network" section between summary stats and Committee Assignments. Three plain-language lines (each guarded by `{% if %}`): (1) "Shares bills with **N legislators** (X Republicans, Y Democrats)", (2) "Their own bills attract **N co-sponsors** on average (Nx the chamber average)", (3) "Bills they join as co-sponsor pass at **Nx** the typical rate (X% vs Y% chamber median)" (only shown when multiplier > 1.0). Wording carefully distinguishes "their own bills" vs "bills they join" to avoid confusion with the top-level cross-party support metric. All numbers shown with transparent baselines — no mystery multipliers. Renamed top-level "Cross-party co-sponsorship" to "Cross-party support: X% of their bills have opposite-party co-sponsors" for clarity. Also replaced "Network centrality: 0.342" in the Moneyball Composite table with "Network reach: N collaborators (0.342)" for contextualised display.
  - **`base.html`:** Added CSS for `.influence-network` (green-left-border box on light green background), `.influence-title`, `.influence-list`, `.influence-detail`. Consistent with existing card section styling.

- **Committee Power Dashboard on advocacy cards:**
  - **`analytics.py`:** Added `CommitteeStats` dataclass (total_bills, advanced_count, passed_count, advancement_rate, pass_rate) and `compute_committee_stats()` function that classifies each committee's bills by pipeline stage and status. Added `build_member_committee_roles()` which builds a reverse index (member_id -> list of committee assignment dicts) with role, leadership flag, committee name, and per-committee bill advancement stats. Roles sorted: Chair > Vice-Chair > Minority Spokesperson > Member.
  - **`main.py` (AppState + lifespan):** Added `state.committee_stats` and `state.member_committee_roles` fields. Computed at startup (Step 3b) after committee data loads. `_member_to_card()` now includes a `committee_roles` list in each card dict, populated from the pre-built reverse index.
  - **`_results_partial.html`:** New collapsible "Committee Assignments (N)" `<details>` block on each card, positioned between summary stats and Legislative Scorecard. Leadership positions (Chair, Vice-Chair, Minority Spokesperson) shown prominently with role badge, plain-language power explanation ("Controls bill hearings and scheduling"), and bill advancement stats ("Advanced 8 of 12 bills (67%) - 3 became law"). Regular memberships listed compactly as comma-separated names.
  - **`base.html`:** Added CSS for `.card-committees`, `.committees-body`, `.committee-row`, `.committee-role-badge` (color-coded: Chair = brown, Vice-Chair = olive, Spokesperson = slate, Member = gray), `.committee-power-note`, `.committee-stats`, `.committee-member-list`. Follows existing collapsible pattern (`.card-scorecard`).
  - **Fix: committee stats showed "0 of N advanced" for all committees.** Two root causes: (1) **Bill number format mismatch** — committee pages list bills without leading zeros (`SB79`) and sometimes with amendment suffixes (`SB228 (SCA1)`), while the bill cache uses zero-padded numbers (`SB0079`). Added `_normalise_committee_bill_number()` to strip suffixes and zero-pad to 4 digits. This fixed 26% of bill lookups (2,658 of 10,105). (2) **ILGA committee bills page only shows currently-pending bills** — bills that already passed through committee are no longer listed. This meant we were only seeing bills stuck at depth 1 ("Assigned to X"). Fix: Added `_build_full_committee_bills()` which scans every bill's `action_history` for "Assigned to"/"Referred to" actions and matches committee names (with fuzzy matching for "X Committee" suffixes and abbreviation variants). This merges historical assignments with the ILGA page data, giving true throughput. Result: Transportation committee went from "0 of 11 (0%)" to "29 of 51 advanced (57%); 29 became law". Overall: 15,182 bills tracked across all committees, 1,063 advanced, 1,057 became law.

- **Scorecard UI added to advocacy cards:**
  - **`_member_to_card()` (main.py):** Now looks up `state.scorecards.get(member.id)` and attaches a `scorecard` dict with template-friendly fields: Lawmaking (laws_filed, laws_passed, law_pass_rate_pct, magnet_score, bridge_pct), Resolutions (resolutions_filed, resolutions_passed, resolution_pass_rate_pct), Overall (total_bills, total_passed, overall_pass_rate_pct, vetoed_count, stuck_count, in_progress_count). Only included when scorecard data exists and the member has at least one bill.
  - **`_results_partial.html`:** Each advocacy card now has a collapsible `<details class="card-scorecard">` block ("Legislative Scorecard") between the summary stats and "Why this target". Three-section table: Lawmaking (HB/SB), Resolutions (HR/SR/HJR/SJR), Overall. Vetoed/stuck/in-progress rows shown only when non-zero. Tooltips on Magnet and Bridge labels.
  - **`base.html`:** Container widened from `max-width: 760px` to `920px` so the main content column (~600px) has enough room for the scorecard table without label wrapping. Added `.card-scorecard`, `.scorecard-body`, `.scorecard-table`, and `.scorecard-section-head` styles (consistent with existing `.how-it-works` / `.formula-table` look).
  - **Width note:** If scorecard tables still wrap on narrow viewports, increase `.container` max-width further or reduce aside width. The 920px value gives ~600px to `.results-main`.

- **Scorecard: Magnet/Bridge definitions, Moneyball breakdown, wider layout:**
  - Magnet and Bridge rows in the scorecard table now have inline definitions: Magnet = "avg co-sponsors per bill -- higher means the legislator attracts more support"; Bridge = "% of bills with at least one cross-party co-sponsor -- measures bipartisan reach".
  - New "Moneyball Composite" section in the scorecard table shows the individual component scores with weights: Passage rate (24%), Pipeline depth (16%), Co-sponsor pull (16%), Cross-party rate (12%), Network centrality (12%), Institutional role (20%). Data passed via new `moneyball` dict in `_member_to_card()`.
  - Container max-width increased from 920px to 1120px for a wider content area.

- **GraphQL unified search query (`search`):**
  - Created `src/ilga_graph/search.py` — in-memory search engine with tiered relevance scoring (exact ID > exact name > prefix > contains name > contains description > secondary fields). Searches Members (name, id, role, party, district, chamber, committees, bio_text), Bills (bill_number, description, synopsis, primary_sponsor, last_action), and Committees (code, name). Returns `SearchHit` dataclasses with `entity_type`, `match_field`, `match_snippet`, `relevance_score`, and the underlying model. No new dependencies.
  - Added to `schema.py`: `SearchEntityType` enum (MEMBER, BILL, COMMITTEE), `SearchResultType` (entity_type, match_field, match_snippet, relevance_score, optional member/bill/committee), `SearchConnection` (items + page_info).
  - New `search(query, entityTypes, offset, limit)` resolver on Query class. Accepts free-text query, optional entity-type filter, and pagination. Member results include scorecard + moneyball via batch loaders.
  - Future: fuzzy matching (Levenshtein), search indexing at startup, vote events and witness slips as searchable entities, autocomplete endpoint.

## Done (this session)

- **Full-text feature caps for reasonable ML runtime (`ml/features.py`):**
  - **Problem:** Full-text TF-IDF used 1000 features and 5000 words per bill, contributing to large feature matrices and long Step 2/3 runs (model comparison and hyperparameter tuning).
  - **Solution:** Configurable upper limits (env-overridable) so we keep full-text signal without massive size:
    - **FULLTEXT_MAX_FEATURES** default **400** (was 1000). Set `ILGA_ML_FULLTEXT_MAX_FEATURES` to override (e.g. `800` for more signal, `200` for faster runs).
    - **FULLTEXT_MAX_TOKENS** default **2000** (was 5000). Set `ILGA_ML_FULLTEXT_MAX_TOKENS` to override (words per bill used for TF-IDF; lower = faster, higher = more content).
  - `build_full_text_features()` now uses these when `max_features`/`max_tokens` are None. Both `build_feature_matrix()` and `build_panel_feature_matrix()` call it without hardcoded values, so the caps apply everywhere. Log line shows "max N features, M tokens/bill" so you can confirm limits in use.

- **ML Step 3 (hyperparameter tuning) no longer stalls for 30+ min (`bill_predictor.py`):**
  - **Problem:** Step 3 "Tuning GradientBoosting hyperparameters" ran 40 × 5-fold = 200 fits with a large param grid; each GradientBoosting fit on ~14k rows can take 10–30+ seconds, so the step appeared stuck for 30+ minutes with no progress output.
  - **Changes:** (1) Reduced `n_iter` from 40 to 20. (2) Trimmed GradientBoosting param grid (fewer `n_estimators`/`max_depth`/`learning_rate`/`subsample`/`min_samples_leaf` options) so tuning finishes in ~5–10 min. (3) Set `verbose=2` on `RandomizedSearchCV` so sklearn prints progress (e.g. "Fitting 5 folds for each of 20 candidates..."). (4) Added `ILGA_ML_SKIP_TUNE=1` to skip tuning entirely and fit the best model once on full training data for fast iteration — use `ILGA_ML_SKIP_TUNE=1 make ml-run`.

- **Smart tiered index scanning (`scrapers/bills.py`):**
  - **Problem:** Every `make scrape` walked all 125 ILGA index pages (~30 min) even when nothing changed. Wasteful for daily runs.
  - **Tiered strategy:** `incremental_bill_scrape()` now auto-decides scan depth based on `scrape_metadata.json` timestamps:
    - **<24h since last scan → SKIP** index entirely. Only re-scrapes bills with recent activity (from cached `last_action_date`). Takes ~seconds.
    - **<7 days since last full scan → TAIL-ONLY** scan. New `tail_only_bill_indexes()` fetches the ILGA `/Legislation` page once to discover doc types, then only checks range pages at or beyond the highest cached bill number per doc type. E.g., if highest SB is 4052, only page 41 of 41 is fetched (not pages 1-40). Takes ~1-2 min vs 30 min.
    - **>7 days or first run → FULL** scan. All 125 pages as before.
    - **`FULL=1` override:** `make scrape FULL=1` forces a full walk regardless of timestamps.
  - **Metadata tracking:** `scrape_metadata.json` now stores `last_full_scan`, `last_tail_scan`, and `highest_bill_per_type` (e.g., `{"SB": 4052, "HB": 5623, "SR": 614}`). Updated on every scrape. Full scans update both timestamps; tail scans update only `last_tail_scan`.
  - **Helper functions:** `_hours_since()` (timestamp age), `_extract_highest_bill_numbers()` (scan cached bills for max number per doc type), `_bill_number_to_int()` (extract numeric suffix), `tail_only_bill_indexes()` (the tail scan implementation).
  - **Re-scrape logic improved:** Recently active bills (last 30 days) are now re-scraped regardless of scan type — even on "skip" runs. Build entries from cached `Bill.status_url` so no index walk is needed.

- **Pipeline resilience + Makefile overhaul:**
  - **Data loss fix (`scrapers/bills.py`):** `incremental_bill_scrape()` and `scrape_all_bills()` now preserve existing `vote_events` and `witness_slips` when re-scraping bills. Previously, re-running `make scrape` would create fresh `Bill` objects with empty arrays and overwrite the cached bill — destroying hours of vote/slip scraping work. Fix: both functions now carry forward `old.vote_events` and `old.witness_slips` from the existing cached bill before overwriting.
  - **Unified pipeline (`scripts/scrape.py`):** Merged the separate `scripts/scrape.py` (members+bills) and `scripts/scrape_votes.py` (votes+slips) into a single pipeline with 4 phases:
    1. **Phase 1+2:** Members + Bill index + Bill details (incremental, preserves vote/slip data)
    2. **Phase 3:** Votes + witness slips for bills that need them (delegates to `scrape_votes.py` as subprocess)
    3. **Phase 4:** Analytics + Obsidian vault export (optional, `--export`)
  - **Makefile simplified:** Replaced 15+ targets with 8 clear ones:
    - `make scrape` — Smart tiered scan (daily ~2 min, auto-decides). Env vars: `FULL=1` (force full walk), `FRESH=1` (nuke cache), `LIMIT=100` (vote/slip bill limit), `WORKERS=10` (parallel workers), `SKIP_VOTES=1` (phases 1-2 only), `EXPORT=1` (include vault export).
    - `make dev` — Serve from cache (dev mode, auto-reload)
    - `make serve` — Serve from cache (prod mode)
    - `make install` — Install project with dev deps
    - `make test` — Run pytest
    - `make lint` / `make lint-fix` — Ruff check + format
    - `make clean` — Remove cache/ and generated files
  - **No backwards compatibility needed.** Old targets are gone. One command for all data: `make scrape`.

- **Vote PDF tally mismatch fix (`scrapers/votes.py`):**
  - **Root cause:** Single-letter middle initials (E, P, N) in committee vote PDFs were being parsed as vote codes (Excused, Present, Nay). Example: `Y Hastings, Michael E Y Martwick, Robert F` → "E" treated as Excused, so Martwick's Y vote was lost. Always manifested as "expected N yeas, got N-1".
  - **Fix:** Extended the `SMART_A` disambiguation pattern to all single-letter codes that can be middle initials: E, P, N, A. Each code uses a negative lookahead — it's only treated as a vote code when NOT immediately followed by another vote code (e.g., `E(?!\s+(?:NV|Y|N|P|A)\s)`). This means "Michael E Y Martwick" → E followed by Y → middle initial, name becomes "Hastings, Michael E". But "E Smith Y Jones" → E followed by "Smith" → real Excused vote.
  - **PDF artifact fix:** Added post-processing to handle `Y NV Mayfield` extraction artifacts where `NV` gets captured as part of a name. Names starting with a vote code prefix are re-split into the correct code + name.
  - **Tested:** 15 committee + floor vote PDFs, all 15 pass (previously 3+ were failing on every committee PDF with middle initials). Regression tests on floor votes (87Y/28N, 55Y) all pass.

- **Vote/slip scraper robustness overhaul + data recovery (`scrape_votes.py`):**
  - **Root cause:** Progress file (`votes_slips_progress.json`) tracked "done" bills by bill_number but didn't verify actual data existed in `bills.json`. After a cache regeneration, all 11,721 bills were marked "done" in progress but had empty `vote_events`/`witness_slips` arrays. Scraper would report "Nothing to do" while startup showed 0 vote events / 0 slips.
  - **Data recovery:** Merged standalone `cache/vote_events.json` (18 records, 8 bills) and `cache/witness_slips.json` (17,440 records, 5 bills) into `bills.json`. Result: 129 vote events across 62 bills, 17,444 witness slips across 9 bills now load correctly at startup.
  - **Data-verified progress:** Progress validation now checks actual `bill.vote_events` / `bill.witness_slips` arrays, not just the progress file. Stale entries (marked "done" but no data) are automatically dropped. New `--verify` flag rebuilds progress entirely from `bills.json` data.
  - **Dual-track progress:** New `checked_no_data` list tracks bills that were scraped but genuinely had no votes/slips (distinguishes "checked and empty" from "never checked"). Prevents re-scraping known-empty bills.
  - **Heuristic skip:** Bills stalled at intro/assignments (only "Filed with Secretary", "First Reading", "Referred to Assignments" actions) are skipped without HTTP requests — saves ~3,800+ round-trips. These bills can't have votes or witness slips.
  - **Batch saves:** `bills.json` now saved every 25 bills (configurable via `--save-interval`) instead of after every single bill. Eliminates writing ~50MB JSON 11,000+ times. Progress file also batched.
  - **Smart ordering:** SB/HB scraped first (most important), then resolutions, AM last. Previously alphabetical (AM came first, SB/HB never reached).
  - **ETA display:** Each progress line now shows estimated time remaining.
  - **Standalone merge:** Scraper auto-merges `vote_events.json` / `witness_slips.json` into bills at startup if they exist.
  - **Makefile:** Added `make scrape-votes-verify` target.
  - **Estimated full scrape:** ~100 minutes for all ~11,400 remaining bills (5 workers, 0.15s delay). Heuristic skip removes ~255 stalled bills instantly.

- **Lint hardening + cleanup (Ruff E501/W293):**
  - Wrapped long metric/help strings and cleaned blank-line whitespace in:
    `scripts/scrape_votes.py`, `src/ilga_graph/metrics_definitions.py`,
    `src/ilga_graph/schema.py`, `src/ilga_graph/scraper.py`,
    `tests/conftest.py`, and `tests/test_scrape_votes_sample.py`.
  - Added commit-time lint guard: new `.pre-commit-config.yaml` with Ruff check
    (`--fix`) and Ruff format hooks.
  - Added `pre-commit` to dev dependencies and Make targets:
    `make hooks` (install hooks) and `make hooks-run` (run on all files).
  - Fixed exporter regression in `exporter.py`: `_render_member()` now falls back
    to member-attached bills (`sponsored_bills` / `co_sponsor_bills`) when no
    global `bills_lookup` is passed, restoring `[[SBxxxx]]` / `[[HBxxxx]]` links
    in direct rendering tests.
  - Strengthened commit gate: `.pre-commit-config.yaml` now runs full
    `make test` in addition to Ruff, blocking commits on failing tests.

- **Incremental votes/slips scraper (`make scrape-votes`):**
  - Created `scripts/scrape_votes.py` -- standalone incremental scraper for roll-call votes and witness slips. Derives bill list from `cache/bills.json` (all bills with `status_url`), not from hardcoded URLs. Progress tracked in `cache/votes_slips_progress.json` so each run picks up where the last left off.
  - **Resumable:** Progress saved after each bill completes (atomic writes). Ctrl+C at any time -- completed bills are already persisted. Run again to continue from next unscraped bill.
  - **Parallel:** Uses `ThreadPoolExecutor` (default 5 workers) for concurrent bill scraping. `--workers N` to tune.
  - **Real-time output:** Each bill prints a progress line as it completes (`[3/10] HB0034 -- 4 votes, 3217 slips (2.1s)`), plus a summary at end or on interrupt.
  - **CLI:** `--limit N` (default 10, 0 = all remaining), `--workers N` (default 5), `--fast` (shorter delay), `--reset` (wipe progress).
  - **Makefile:** `make scrape-votes` (next 10), `make scrape-votes LIMIT=50`, `make scrape-votes LIMIT=0` (all).
  - **scrape.py cleanup:** Removed inline votes/slips scraping from `scripts/scrape.py`. Bill scrape now preserves existing per-bill `vote_events`/`witness_slips` in cache (no wipe). Log message points to `make scrape-votes`.

- **Roll-call votes / witness slips count verified (not a display bug):**
  - Startup table now shows bill coverage: e.g. "13 vote events (5 bills)" and "17440 slips (5 bills)" so it's clear data is from a subset.

- **Fixed "laws passed: 0 of 0 filed" — all members showing accurate bill stats now.**


- **Fixed "laws passed: 0 of 0 filed" bug (two root causes):**
  - **`is_shell_bill()` false positive** (`analytics.py`): The `len(desc) < 50` threshold was marking **every single bill** as a "shell bill" because ILGA index descriptions are abbreviated titles (max 30 chars, median 25). For example, "CRIM CD-FIREARM SILENCER" (24 chars) was flagged as a shell bill. Fix: Removed the length threshold entirely. Now only matches keyword "Technical", "Shell", or the `-TECH` suffix (ILGA abbreviation for technical/procedural bills like "LOCAL GOVERNMENT-TECH"). This correctly identifies ~1 true shell bill per ~70 substantive bills.
  - **Stale analytics cache** (`analytics_cache.py`): The freshness check (`_member_cache_mtime`) only compared `scorecards.json`/`moneyball.json` against `members.json` mtime. When the full scrape updated `bills.json` (with `sponsor_ids` populated), the analytics cache wasn't invalidated because `members.json` hadn't changed. Fix: Renamed to `_source_data_mtime()` and now checks the **max** mtime of both `members.json` AND `bills.json`. Deleted stale cache files so analytics recompute on next server start.
  - **Verified**: 179 of 180 members now show accurate `laws_filed` / `laws_passed` / `passage_rate`. Totals: 7,126 laws filed, 419 passed across all members. Sample: Neil Anderson 66 filed / 1 passed (1.5%), Omar Aquino 26/2 (7.7%), Christopher Belt 56/4 (7.1%).



- **Discovery-based scraping: parse /Legislation to find ALL doc types and exact ranges:**
  - **New approach** (`scrapers/bills.py`): Added `DocTypeInfo` dataclass and `_discover_doc_types()`. Fetches the ILGA `/Legislation` page once and parses ALL doc type sections (SB, HB, SR, HR, SJR, HJR, SJRCA, HJRCA, EO, JSR, AM) and their exact range page URLs. This eliminates blind pagination, the single-page optimization hack, and the recycled-data guard.
  - **Refactored `scrape_bill_index()`**: Now accepts an optional `range_urls` parameter (from discovery). When provided, iterates over the exact known URLs — no guessing, no infinite loops. Falls back to blind pagination with recycled-data guard only when range_urls is not provided.
  - **Refactored `scrape_all_bill_indexes()`**: Calls `_discover_doc_types()` first, then iterates over all discovered types. SB/HB respect existing `sb_limit`/`hb_limit` CLI flags. All other types are always scraped in full (they're small).
  - **Removed dead code**: `_single_page_list_url()` and `_scrape_bill_index_single_page()` are gone — replaced by the discovery approach.
  - **Chamber derivation fix**: `scrape_bill_status()` now correctly assigns `chamber = "J"` (Joint/Executive) for EO, JSR, and AM doc types, instead of incorrectly defaulting to "H".
  - **Verified**: Full isolated test discovered 11 doc types (125 range pages) and scraped **~15,000+ entries** across all categories:
    - SB: ~4052 | HB: 5623 | SR: 614 | HR: 660 | SJR: 52 | HJR: 54
    - SJRCA: 11 | HJRCA: 25 | EO: 8 | JSR: 1 | AM: 621
  - **Backward compatible**: `sb_limit`/`hb_limit` CLI flags, `etl.py`, `scripts/scrape.py`, and all downstream code unchanged.

- **Previously fixed: infinite loop + 200-bill truncation (now superseded by discovery):**
  - The infinite loop bug (ILGA returning recycled SB0001-SB0100 for out-of-range queries) and the 200-bill truncation bug (single-page optimization returning only 100 per chamber) are both eliminated by the discovery approach. The recycled-data guard is kept only as a safety net in the fallback path.

- **Separation of concerns + cache simplification + single-page index (full plan execution):**
  - **Bill model** (`models.py`): Added `vote_events: list[VoteEvent]` and `witness_slips: list[WitnessSlip]` fields to `Bill` dataclass. Each bill now carries its own vote and slip data.
  - **Single-page SB/HB index** (`scrapers/bills.py`): Added `_single_page_list_url()` and `_scrape_bill_index_single_page()`. For SB and HB, the scraper now fetches the full bill list in **one request per chamber** instead of ~85 paginated requests. Falls back to range-based pagination if the single-page request fails.
  - **Bills cache** (`scrapers/bills.py`): `_bill_from_dict()` and `_bill_to_dict()` now serialize/deserialize `vote_events` and `witness_slips` per bill. `save_bill_cache()` uses atomic writes (temp file + rename).
  - **Unified committees.json** (`scraper.py`): Merged `committee_rosters.json` and `committee_bills.json` into a single `committees.json` where each committee entry includes `roster` and `bill_numbers` inline. Separate cache files no longer written. `_save_unified_committee_cache()` replaces `_save_split_committee_cache()`. Atomic writes used.
  - **GraphQL schema** (`schema.py`): Added `ActionEntryType` (date, chamber, action) and `action_history: list[ActionEntryType]` to `BillType`. **MemberType** now exposes `sponsored_bill_ids: list[str]` and `co_sponsor_bill_ids: list[str]` instead of embedded `sponsored_bills: list[BillType]` / `co_sponsor_bills: list[BillType]`. Separation of concerns enforced at the API level.
  - **Exporter** (`exporter.py`): Bill notes now include an `## Actions` table (date, chamber, action) from `bill.action_history`. Member notes use `member.sponsored_bill_ids` / `member.co_sponsor_bill_ids` + a bills lookup to render `[[bill_number]]` links (IDs only, no embedded bills). Bill set built exclusively from `all_bills` parameter. Co-sponsors resolved by iterating member `co_sponsor_bill_ids`.
  - **ETL + main.py**: `state.vote_events`, `state.vote_lookup`, `state.witness_slips`, `state.witness_slips_lookup` now populated from per-bill data in `state.bills` (no separate vote/slip scraping in the API startup). Resolvers unchanged — they still read from `state.vote_lookup` and `state.witness_slips_lookup`.
  - **Scrape script** (`scripts/scrape.py`): After scraping votes/slips, merges them into the corresponding `Bill` objects in `data.bills_lookup` and re-saves `bills.json` with atomic write. No separate `vote_events.json` or `witness_slips.json` files.
  - **Cache layout** (new minimal set):
    - `members.json` — all members, member details only (IDs for bill references)
    - `committees.json` — all committees with roster + bill_numbers inline
    - `bills.json` — heavyweight: each bill has status, action_history, vote_events, witness_slips
    - `zip_to_district.json` — ZIP → district mapping (unchanged)
    - `scorecards.json` / `moneyball.json` — optional derived cache (recomputed if stale)
    - `scrape_metadata.json` / `bill_index_checkpoint.json` — transient operational files

## Previously done

- **Bill index scraping improvements:** Fixed two issues with bill index scraping: (1) **Clearer progress logs**: Terminal now shows expected bill number ranges using standard convention (e.g., "SB 0001-0100 (page 1)", "SB 0101-0200 (page 2)") and reports actual bills found (e.g., "✓ Parsed 42 bills: SB0001 to SB0042"). This makes it clear which 100-bill range you're on and how many bills actually exist. (2) **Incremental checkpoint saving**: Bill index scraping now saves progress to `cache/bill_index_checkpoint.json` after every page (~100 bills). If interrupted (Ctrl+C), rerunning the scrape will resume from the checkpoint instead of starting over. Checkpoint is cleared automatically when scraping completes successfully. This applies to both the index phase and the detail scraping phase (which already had checkpointing every 50 bills).

- **Chamber-specific associated member fields:** Previously, member associations were generic "Associated Members" for both senators and representatives, which was confusing since senators have 2 representatives and representatives have 1 senator. Changes: (1) Replaced `associated_members` with two chamber-specific fields in GraphQL `MemberType`: `associated_senator` (for Representatives - singular) and `associated_representatives` (for Senators - plural). These are computed from the underlying `associated_members` data based on chamber. (2) Updated `exporter.py` to render chamber-specific section headers in Obsidian markdown: "## Associated Representatives" for senators, "## Associated Senator" for representatives. This makes the API more semantically clear and the UI more intuitive.

- **Full bill index + scrape-200:** Bill index previously only fetched one range page (100 SB + 100 HB). Implemented pagination in `scrape_bill_index()`: loop over range pages (0001-0100, 0101-0200, ...) until we have enough or a page is empty. `limit=0` means all pages (~9600+ bills). Added **make scrape-200** (200 SB + 200 HB, 2 range pages per type) to test pagination and **make scrape-full** (--sb-limit 0 --hb-limit 0) for full index. README and scrape.py docstring updated. Full scrape is slow (many index pages + one BillStatus request per bill).

- **Scrape → dev/prod pipeline:** Simplified to a clear two-step flow: (1) scrape data into cache (choose size), (2) serve via API. **make scrape** = full scrape for prod (all members, 300 SB + 300 HB). **make scrape-dev** = light scrape for dev (20/chamber, 100 SB + 100 HB, fast). **make dev** / **make dev-full** / **make run** now set **ILGA_LOAD_ONLY=1** so the server only loads from cache (no scraping on startup). Removed **make scrape-fast** in favor of **make scrape-dev**. README and scrape.py docstring updated; .env.example documents ILGA_LOAD_ONLY.

- **Evidence-based script hints & "How we pick these targets":** Script hints were hardcoded generic strings ("high effectiveness, deep network connections") that assumed the user already understood the system. A first-time visitor had no way to know *why* a legislator was chosen or what "effectiveness" meant. Changes: (1) Added `_build_script_hint_*` helpers in `main.py` that inject real numbers (laws passed X of Y, Z% passage rate, W% cross-party) into each card's script hint. Senator/Rep hints include ZIP and district. Power Broker hint explains Chair vs Moneyball selection with stats. Ally hint cites cross-party %. Super Ally merges both with evidence. (2) Template macro `member_card` now reads `member.script_hint` from the card dict instead of receiving a hardcoded 4th argument. (3) Added collapsible "How we pick these targets" section at the bottom of results: plain-language definitions of Senator/Rep (Census ZIP match), Power Broker (committee chair or highest Moneyball), Potential Ally (seatmate with highest cross-party rate), and Super Ally (merged). (4) Nested collapsible "How is the Moneyball score calculated?" with a table of the 6 components, weights, and what each measures. (5) Added CSS for `.how-it-works` collapsible and `.formula-table` in `base.html`. No algorithm changes — only explanation and evidence in the UI.

- **Scorecard / Moneyball clarity:** User wanted empirical stats (bills passed, vetoed, passage rate) front and center and derived metrics (Moneyball, effectiveness) clearly defined to avoid vagueness and bloat. (1) Added `src/ilga_graph/metrics_definitions.py` as single source of truth: empirical metric definitions (laws filed/passed, passage rate, magnet, bridge, pipeline, centrality, etc.) and Moneyball formula (one-liner + component list with weights). (2) Advocacy cards now show empirical first: "Laws passed: X of Y (Z% passage)", "Cross-party co-sponsorship: W%", then "Moneyball: N (composite rank 0–100)" with a "?" tooltip that shows the one-liner. (3) GraphQL query `metricsGlossary` returns full glossary (empirical, effectiveness_score, moneyball_one_liner, moneyball_components) so any client can render tooltips or docs. (4) README section "Metrics: empirical vs derived" documents the approach. No new metrics added; we only clarified and reordered display.

- **Advocacy full-data / seed-mode clarity:** Banner was always showing "Only 40 of 177 legislators" and "~14 demo ZIPs" even when full cache was loaded. Root cause: `make dev` sets `ILGA_SEED_MODE=1`, so (1) members load from cache when present (so user can have 177 after `make scrape`), but (2) ZIP crosswalk always used hardcoded seed (~14 ZIPs) when seed mode ON. Changes: (1) Advocacy banner now uses actual `member_count` and `zip_count` and only shows "Only N of 177" when N < 100; explains how to get full data (`make run` or `ILGA_SEED_MODE=0`). (2) Added `make dev-full` (dev profile + `ILGA_SEED_MODE=0`) to run with full cache + full Census ZCTA for testing. (3) Documented in README ("Testing with full data") and .env.example. To test advocacy with all members and all ZIPs: `make scrape` then `make dev-full` or `make run`.

---

## Next (when you're ready)

- **Power Map enhancements:**
  - Weighted edges (shared bill count) for better pruning and edge thickness visualization.
  - ~~Fix influence engine~~ — **DONE:** `compute_influence_scores()` now receives `member_lookup_by_id` (id-keyed) instead of the name-keyed `member_lookup`. Influence scores, leaderboard, Power Movers, and graph node sizing all use real influence data now.
  - House committee mapping in `_CATEGORY_COMMITTEES` (currently Senate only).
  - Free-text topic search (type "kei truck" → match bills → find which committees they're assigned to → highlight those members).
  - Mobile/responsive: detail panel as bottom sheet on small screens.
  - Path visualization: when topic + ZIP are both set, draw dotted paths from user's legislators to committee chairs.
  - Time-based animation: show how co-sponsorship network evolves over the legislative session.

- **Run full pipeline** (PRIORITY): Pipeline is fixed and unified with smart tiered scanning.
  - `make scrape` — Daily run (~2 min if <24h since last scan, auto-decides tier).
  - `make scrape FULL=1` — Force full index walk (all 125 pages, ~30 min). Use weekly or after major ILGA updates.
  - `make scrape WORKERS=10` — More parallel workers for votes/slips.
  - `make scrape LIMIT=100` — Limit vote/slip phase to 100 bills (for testing).
  - `make scrape FRESH=1` — Nuke cache and re-scrape from scratch.
  - Can Ctrl+C and resume at any time. Batch saves every 25 bills. Vote/slip data is never lost by re-scraping bills.
  
- **Verify GraphQL:** After scrape completes, `make dev` and check that scorecards/Moneyball recompute with vote data. Spot-check 3-4 members' "Laws passed: X of Y" against ilga.gov to verify accuracy. Test advocacy frontend with real ZIPs to see if Power Broker / Ally selections make sense with real vote/slip data.

- When shifting to prod: set `ILGA_PROFILE=prod`, `ILGA_CORS_ORIGINS`, and optionally `ILGA_API_KEY`.

---

## Production (checklist for deployment)

- Pre-populate `cache/` (full or incremental scrape) so startup is load-only.
- Set `ILGA_PROFILE=prod` (turns off dev mode + seed mode automatically).
- Set `ILGA_CORS_ORIGINS` to your front-end origin(s) — prod warns if unset.
- Set `ILGA_API_KEY` to protect the GraphQL endpoint — prod warns if empty.
- Use `GET /health` for readiness (`ready` is true when members are loaded).

---

## ML Pipeline (`feature/ml-pipeline` branch)

**New `src/ilga_graph/ml/` package** — fully automated "Legislative Intelligence Engine" (v2: robust training). One command (`make ml-run`) transforms raw cached data into enriched analytics. **No interaction required** -- the ML teaches itself from the data.

### v2 improvements (what changed)

| Problem in v1 | Fix in v2 |
|---|---|
| Test set had 3 positive / 2900 negative (broken evaluation) | Only evaluates on "mature" bills (120+ days old). Test set now 412 advanced / 1650 stuck -- real metrics. |
| Single algorithm, no validation | Compares 4 algorithms (GradientBoosting, RandomForest, LogisticRegression, AdaBoost) with 5-fold stratified cross-validation. Best model auto-selected. |
| No hyperparameter tuning | RandomizedSearchCV (40 iterations) tunes the winner. Best: GBT with `n_estimators=300, max_depth=9, lr=0.05`. |
| No probability calibration | Isotonic calibration via `CalibratedClassifierCV` -- probabilities are now reliable, not just rankings. |
| 49% of legislators unclassified (DBSCAN outliers) | Switched to Agglomerative Clustering -- **100% members classified** (0% outliers). Optimal k auto-selected (k=10). |
| Anomaly detector flagged genuine controversy (big = suspicious) | New coordination features (name duplication rate, position unanimity, top org share). Now explains WHY each bill flagged. |
| No quality report | Generates `model_quality.json` with trust assessment, comparison table, strengths/issues. |

### Current results (v2)

| Metric | Value |
|---|---|
| **Bill prediction: CV ROC-AUC** | 0.984 +/- 0.005 (5-fold, GradientBoosting) |
| **Bill prediction: Test ROC-AUC** | 0.910 (held-out, calibrated) |
| **Bill prediction: Test precision (advanced)** | 96.3% (when it says "advance", it's right) |
| **Bill prediction: Accuracy on mature bills** | 94.9% |
| **Entity resolution** | 100% (385/385 unique names) |
| **Coalition clustering** | 10 blocs, 100% members classified, 100% cross-party blocs |
| **Anomaly detection** | 102 bills flagged (8%) with coordination reasons |
| **Pipeline runtime** | ~4 minutes (was 21s but now does proper CV + tuning) |

### What it produces (in `processed/`)

| Output file | What it is |
|---|---|
| `bill_scores.parquet` | Every bill scored with probability of advancement. Mature bills have reliable labels; immature bills are true forecasts. |
| `model_quality.json` | Trust assessment: model comparison, test metrics, strengths/issues, top features. |
| `coalitions.parquet` | Every legislator assigned to a voting bloc (100% classified, no outliers). |
| `member_embeddings.parquet` | 32-dimensional vector per legislator (spectral graph embeddings). |
| `slip_anomalies.parquet` | Bills scored for coordination signals with human-readable reasons. |
| `fact_vote_casts.parquet` | Vote casts with member_id FK (100% entity resolution). |
| `dim_*.parquet`, `fact_*.parquet` | Normalized star schema tables. |

### How to run

```bash
make ml-setup    # Install ML dependencies (one time)
make ml-run      # Run full pipeline (~4 minutes, no interaction)
```

### Pipeline steps (all automated)

1. **Data Pipeline** (`ml/pipeline.py`) — Flattens `cache/*.json` into 6 Parquet tables (726K+ rows, ~7s).
2. **Entity Resolution** (`ml/entity_resolution.py`, `ml/active_learner.py`) — Maps 385 unique vote-PDF names to member IDs. **100% resolved** (98.7% exact match + gold mappings). No human input needed.
3. **Bill Outcome Prediction** (`ml/features.py`, `ml/bill_predictor.py`) — Compares 4 algorithms with 5-fold stratified CV, tunes hyperparameters (40 iterations), calibrates probabilities, evaluates on mature bills only (120+ day maturity threshold). Scores all 9,676 bills. **CV AUC: 0.984, Test AUC: 0.910**. Top features: text signals ("makes technical"), intro timing, sponsor count, witness slip support. Time-based split on mature bills prevents leakage.
4. **Coalition Discovery** (`ml/coalitions.py`) — Builds agreement-rate graph (normalized for activity, not raw counts) + co-sponsorship. Agglomerative clustering with auto-k selection (silhouette analysis, k=3..10). **10 blocs, 100% members classified, 100% cross-party.**
5. **Anomaly Detection** (`ml/anomaly_detection.py`) — Isolation Forest on coordination features (name duplication, position unanimity, org concentration, top org share). Flags 102 bills (8%) with human-readable reasons (e.g., "single org files 81% of slips; near-unanimous position").

### Individual steps (optional)

```bash
make ml-pipeline   # Data pipeline only
make ml-resolve    # Entity resolution only (AUTO=1 for non-interactive)
make ml-predict    # Bill scoring only
```

- **Dependencies**: `polars`, `pyarrow`, `scikit-learn`, `rapidfuzz`, `networkx`, `rich` in `[project.optional-dependencies] ml`.

### v3: Intelligence Dashboard & Self-Correcting Feedback Loop

- **Backtester module** (`ml/backtester.py`): Snapshots predictions after each run, backtests previous predictions against new actual outcomes on the next run, accumulates accuracy history in `processed/accuracy_history.json`. Tracks precision, recall, F1, confidence calibration, and biggest misses per run.
- **Chained pipeline** (`Makefile`): `make scrape` now automatically triggers `make ml-run` at the end, so new data is immediately processed and predictions backtested.
- **ML data loader** (`ml_loader.py`): Reads all ML outputs (bill scores, coalitions, anomalies, model quality, accuracy history) into typed dataclasses for API consumption. Loaded at app startup into `state.ml`.
- **GraphQL API**: 7 new resolvers — `billPredictions` (filterable by outcome, confidence, reliability, forecasts), `billPrediction` (single bill), `votingCoalitions` (grouped with party breakdown), `slipAnomalies` (filterable, sorted by score), `modelQuality` (trust assessment, metrics, feature importances, model comparison), `predictionAccuracy` (accuracy history across runs).
- **Web dashboard** (`/intelligence`): HTMX tabbed interface with 4 views:
  - **Predictions tab**: All bill scores with inline score bars, client-side filters (forecasts only, ADVANCE/STUCK, high confidence)
  - **Coalitions tab**: Voting bloc cards with party composition bars, cross-party highlights, expandable member lists
  - **Anomalies tab**: Flagged bills with coordination signal details (org HHI, position unanimity, top org share)
  - **Model Quality & Accuracy tab**: Model comparison, trust assessment, feature importances, accuracy history trend, biggest misses
- **Navigation**: Links between `/advocacy` and `/intelligence` dashboards.
- **Pipeline steps updated** (7 steps): 0=Backtest, 1=Data Pipeline, 2=Entity Resolution, 3=Bill Scoring, 4=Coalitions, 5=Anomaly Detection, 6=Snapshot

### v4: Coalition Naming, Bill Pipeline Stage, Stuck-Bill Analysis

- **Coalition characterization** (`ml/coalitions.py`): After clustering, new `characterize_coalitions()` joins votes → bills → committees to profile each bloc by policy focus. Generates descriptive names from partisan composition + policy areas + voting style + cohesion (e.g., "Consensus Dem-Leaning Education & State Government Bloc", "GOP-Leaning Education & State Government Bloc"). Computes per-coalition: top 3 policy areas, YES rate, cohesion score, and 5 signature bills (highest YES support). Saves `coalition_profiles.json`.
- **Bill pipeline stage** (`ml/features.py`): New `compute_bill_stage()` walks each bill's action history to determine the highest legislative stage reached (FILED → IN_COMMITTEE → PASSED_COMMITTEE → FLOOR_VOTE → CROSSED_CHAMBERS → PASSED_BOTH → SIGNED, or VETOED). Returns stage name + progress fraction (0.0–1.0).
- **Stuck-bill analysis** (`ml/features.py`): New `classify_stuck_status()` sub-classifies non-advancing bills into 5 nuanced statuses: DEAD (vetoed/tabled/session-dead), STAGNANT (180+ days inactive), SLOW (60–180 days), PENDING (active within 60 days), NEW (introduced <30 days ago). Each includes a human-readable reason string.
- **Score columns updated** (`ml/bill_predictor.py`): `score_all_bills()` now replaces `actual_outcome` with `current_stage`, `stage_progress`, `stage_label`, `days_since_action`, `stuck_status`, `stuck_reason`. Uses `last_action_date` from dim_bills for staleness.
- **ML loader updated** (`ml_loader.py`): `BillScore` dataclass gets 8 new stage/stuck fields. `CoalitionMember` gets `coalition_name` and `coalition_focus`. New `CoalitionProfile` dataclass. `MLData` includes `coalition_profiles` list.
- **GraphQL API** (`main.py`): `BillPredictionType` gains `currentStage`, `stageProgress`, `stageLabel`, `daysSinceAction`, `stuckStatus`, `stuckReason`. Bill predictions resolver gains `stuckStatus` and `stage` filters. `CoalitionGroupType` gains `name`, `focusAreas`, `yesRate`, `cohesion`, `signatureBills`. New `SignatureBillType`.
- **Predictions dashboard** (`_intelligence_predictions.html`): Replaced "Actual" column with pipeline progress bar (color-coded by stage), stuck-status badge (color-coded: red=stagnant, orange=slow, yellow=pending, blue=new, gray=dead), and days-idle indicator. Added client-side filters for stuck sub-statuses (Stagnant, Dead, Slow, New).
- **Coalitions dashboard** (`_intelligence_coalitions.html`): Replaced "Coalition N" labels with generated names. Added policy focus tags, cohesion/YES-rate stats, signature bills table per coalition card.
- **CSS** (`base.html`): Pipeline progress bars, stuck-status badge colors, days-idle indicators, focus tags, coalition stats.

- **Next (ML backlog)**:
  - **DONE: True Influence Engine** — betweenness centrality, vote pivotality, sponsor pull, unified InfluenceScore (0–100). Dashboard tab, GraphQL query, coalition enrichment.
  - **DONE: Prediction Table v5** — fixed vetoed-as-ADVANCE bug, lifecycle status (OPEN/PASSED/VETOED/DEAD), staleness features (7 new model inputs), prediction override for terminal bills, sortable columns.
  - **DONE: ML Pipeline Bug Audit** — TF-IDF zero-vector fallback, KeyError guards, dynamic majority party, sponsor rate null guard, vote ties, anomaly NaN guard, defensive date-parse logging.
  - **DONE: Intelligence Story Redesign** — narrative executive summary at `/intelligence`, member deep-dive at `/intelligence/member/:id`, bill deep-dive at `/intelligence/bill/:id`, raw tables at `/intelligence/raw`.
  - Individual vote prediction (recommender system using member embeddings from matrix factorization)
  - "Poison pill" detector (semantic similarity between original and amendment synopses)
  - Committee assignment prediction (multi-class text classifier)
  - Accuracy trend visualization (sparklines or mini chart in accuracy tab)
  - Per-bill prediction history (track how confidence changes over time)
  - Influence trend tracking (snapshot influence scores per run, like backtest does for predictions)
  - **Gold labels (`processed/bill_labels_gold.json`):** Maps bill `leg_id` → 0 (stuck) or 1 (advanced). Human-corrected outcomes for bills the model gets wrong. Possible uses: (1) **Eval only** — report accuracy on gold set in `model_quality.json` without changing training; (2) **Override training** — use gold labels for those bills in the target vector so the model learns from corrections; (3) **Both**. Currently 9 entries, all label 0. Add or edit entries as you find model mistakes. Not yet wired into the pipeline — implement when ready.
  - **Low-confidence UX:** Predictions table now marks rows with confidence <70% as "Uncertain" (amber border + label). Filter checkbox added. Future: expose calibration curve in accuracy tab; per-bill confidence trend over pipeline runs.

---

## Backlog / Future

- (Done: unified GraphQL `search` query — cross-entity free-text search with relevance scoring.)
- Search enhancements: fuzzy matching (Levenshtein), startup index for O(1) lookups, vote events / witness slips as searchable entities, autocomplete / query suggestions endpoint, TF-IDF weighting if data grows.
- (Done: full bill index via `make scrape-full` — pagination in bills.py; 0 = all pages.)
- (Done: `make scrape-votes` derives vote/slip bill list from cache for incremental coverage.)
- Advocacy frontend: add member photos, richer script text, email links.
- Advocacy frontend: move embedded CSS to `static/style.css` when it grows.
- Advocacy frontend: htmx-powered "drill down" on each card (click to expand full member profile inline).
- Advocacy frontend: interactive map visualization — all senators plotted by ZIP/district, filterable by policy category. Click a district to see its advocacy targets.
- Full Census crosswalk download for prod (currently seed-mode hardcoded ZIPs in dev).

---

## Done (summary)

**Modularity Roadmap Steps 1-2 (config + resilience):** Created `src/ilga_graph/config.py` with `ILGA_PROFILE=dev|prod` (one-knob environment switching) + `.env` support via `python-dotenv`. Profile sets sensible defaults; individual vars override. Prod profile warns at startup if CORS or API_KEY are unconfigured. Updated `bills.py`, `votes.py`, `witness_slips.py`, `scraper.py`, and `main.py` to import from config (removed all hardcoded GaId/SessionId). Added try/except around every lifespan ETL step with stale-cache fallback. Added checkpoint saves (every 50 bills) in `scrape_all_bills`. Makefile uses `ILGA_PROFILE=dev`/`prod` instead of inline flags.

**Cache & seed:** Committee data in `cache/committees.json`; seed data in `mocks/dev/` with `ILGA_SEED_MODE=1` for instant dev; normalized `members.json` + `bills.json` (~70% size reduction); vote events and witness slips cached.

**ETL:** Composable steps — `load_or_scrape_data()`, `compute_analytics()`, `export_vault()`; CLI `scripts/scrape.py` can run each step or full pipeline.

**Resilience & performance:** HTTPAdapter with retries and connection pooling; stale cache warnings; startup timing logs and `.startup_timings.csv`; startup summary table now uses chronological ETL-oriented phases with explicit per-step times/details (core load, analytics, seating, vault export, committee indexes, vote index + normalization, member voting records, witness slips, ZIP crosswalk); CSV includes seating_s, slips_s, zip_s, votes, slips, zctas; vote/slip cache eliminates re-scraping on every start.

- (Done) Add `scripts/startup_timings_report.py`: rich/animated CLI to analyze `.startup_timings.csv` across schema versions and show recent-vs-baseline regressions (`make startup-report`).
- (Done) Clarify full-cache runtime mode: `make dev-full` now sets `ILGA_DEV_MODE=0` so startup does not imply 20/chamber + top-100 caps, and startup logs now explicitly distinguish cache-only vs scrape startup.
- (Done) Rework startup summary table into chronological ETL-oriented phases with clearer per-step timing/details (adds committee-index and member-voting-record steps so timing output maps to actual runtime order).

**Witness slips & GraphQL:** Slips and votes share bill list (`ILGA_VOTE_BILL_URLS`); witness slip summary and paginated summaries; `billSlipAnalytics`, `memberSlipAlignment`; advancement analytics (`billAdvancementAnalyticsSummary`); docs in `graphql/README.md` and main README.

**Bills-first & incremental:** Bills from Legislation pages (single source of truth); bill index to BillStatus detail; `scrapers/bills.py` with `incremental_bill_scrape()`; member-bill linkage from `Bill.sponsor_ids`; cache persisted every run; exporter uses `all_bills` from `bills_lookup` so all cached bills get vault notes.

**Server & CI:** Single app in `main.py` (schema.py types only); ruff lint/format and per-file ignores for schema; tests updated for line-length and duplicate keys.

**Seating Chart / Whisper Network (Verdict 1c):** Added `seat_block_id`, `seat_ring`, `seatmate_names`, `seatmate_affinity` fields to `Member` dataclass. Created `src/ilga_graph/seating.py` with fuzzy name matching (bare last name, "Initial. Lastname", compound last names like "Glowiak Hilton"), the Aisle Rule (neighbors within same section only, no cross-aisle adjacency), and co-sponsorship affinity calculation (overlap % of member's bills with seatmate co-sponsors). Exposed all four fields via GraphQL `MemberType`. Integrated into lifespan (Step 2b) and `run_etl`. Senate-only for now; seating data loaded from `mocks/dev/senate_seats.json`.

**SSR Advocacy Frontend:** Added `jinja2` and `python-multipart` dependencies. Created `src/ilga_graph/zip_crosswalk.py` — Python port of the Google Sheets Census crosswalk. Seed-mode fallback has ~14 ZIPs (6 produce full 4-card results with the 40-member dev dataset). Jinja2 templates with htmx (v2.0.4 CDN) for in-page search. `GET /advocacy` renders the search form; `POST /advocacy/search` returns up to 4 cards: **Your Senator**, **Your Representative**, **Power Broker**, and **Potential Ally**. Each card has a "Why this target" box explaining the analytics reasoning and a role-specific script hint. **Super Ally merge:** when Power Broker and Ally resolve to the same person, they collapse into a single "Super Ally" card with both badges and a combined explanation. **Policy category filter:** optional dropdown maps 12 policy areas (Transportation, Education, etc.) to Senate committee codes; when selected, Power Broker and Ally are restricted to members on those committees (with graceful fallback if the filter eliminates all candidates). Committee rosters loaded from `state.committee_rosters`. Results split into "Your Legislators" and "Strategic Targets" sections.

**Steps 3–5 (ETL, analytics cache, DataLoaders):** **(Step 3)** Created `src/ilga_graph/etl.py` with `load_from_cache()`, `load_or_scrape_data()`, `compute_analytics()`, `export_vault()`, `run_etl()`, and `load_stale_cache_fallback()`. Added `ILGA_LOAD_ONLY`: when set, API startup only loads from cache (no scrape). Lifespan uses load-only path when `LOAD_ONLY=1`; scrape script uses etl and supports `--export-only` with `load_from_cache()`. **(Step 4)** Added `analytics_cache.py`: `save_analytics_cache()` / `load_analytics_cache()` persist scorecards and Moneyball to `cache/scorecards.json` and `cache/moneyball.json`. Staleness: cache is used only if newer than member data (`members.json` mtime). Lifespan and scrape script load when fresh, else compute and save. **(Step 5)** Added `loaders.py` with `ScorecardLoader`, `MoneyballProfileLoader`, `BillLoader`, `MemberLoader` (each with `load()` and `batch_load()`). GraphQL context_getter returns `create_loaders(state)`. Resolvers `member`, `members`, and `moneyball_leaderboard` use `info.context` and batch_load for scorecard/profile so list queries do one batched fetch instead of N lookups.

**Moneyball v2 Recalibration:** Three changes to stop undervaluing legislative leadership. **(1) Shell Bill Filter:** `is_shell_bill()` in `analytics.py` identifies procedural placeholders (description contains "Technical"/"Shell" or ends with "-TECH") and excludes them from the effectiveness denominator (`law_heat_score`), so leadership who file shell bills aren't penalised for "failures" that were never meant to pass. Applied in `compute_scorecard()`, `compute_all_scorecards()`, and `avg_pipeline_depth()`. **(2) Institutional Power Bonus:** Added `roles: list[str]` to `Member` (aggregates profile title + committee roster titles via `populate_member_roles()`). New `compute_institutional_weight()` returns 1.0 (President/Leader/Speaker), 0.5 (Chair/Spokesperson), or 0.25 (Whip/Caucus Chair). Added to `MoneyballProfile` as `institutional_weight` and to the composite score at 20% weight (existing weights scaled to 80%: effectiveness 24%, pipeline 16%, magnet 16%, bridge 12%, centrality 12%). **(3) Committee-Chair-First Power Broker:** `_find_power_broker()` now prioritises Committee Chairs of the relevant committee when a policy category is selected, falling back to highest Moneyball score only when no Chair is found. Serialization updated in `scraper.py`; test fixtures updated with realistic 50+ char descriptions.
