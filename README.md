# Predicting Chess Game Outcomes (Chess.com Take-Home)

This project predicts pre-game outcomes from White's perspective:

- `loss`
- `draw`
- `win`

Predictions are performed on March-2026 Titled Tuesday tournaments. Models were trained on February-2026 Titled Tuesday tournaments data and the participating player statistics data. Games data from prior to the actual game in March is also used, making sure data-leakage is avoided.

Chess rating difference is the most important feature to predict expected game result between two player. However, for any individual game, two wain challenges prevent accurate prediction of game results:  

1. Class imbalance: `draw` is rare.  As a consequence, model training loss and selection prioritizes class-fair metrics.
2. Results between similarly rated players has high variance: in particular, the predictability of draw vs non-draw (win/loss) is low.

## Environment Setup

First of all, it is necessary to set up the environment for the project.  
You may use either conda, mamba or micromamba.

From the repository root (using conda):

```bash
conda env create -f environment.yml
conda activate chesscomint
python -m ipykernel install --user --name chesscomint --display-name "Python (chesscomint)"
```

## Dataset Extraction

Use `src/0_data_preparation.ipynb` for runnable bash commands.

Equivalent terminal command from repo root:

```bash
PYTHONPATH=src python3 -m chesscomint.fetch_data \
  --user-agent "chesscomint-interview-exercise"
```

This writes:

- `data/raw/titled_tuesday_games.jsonl`
- `data/raw/titled_tuesday_players.jsonl`

### Data preparation explanation

`src/0_data_preparation.ipynb` is the data-ingestion notebook and is shell-driven for reproducibility.

- It creates required folders (`data/raw`, `data/processed`) so downstream notebooks run without manual setup.
- It runs `src/chesscomint/fetch_data.py`, which:
  - traverses the Chess.com tournament -> round/group endpoints,
  - writes one JSONL row per game to `titled_tuesday_games.jsonl`,
  - extracts unique players and fetches profile/stats payloads into `titled_tuesday_players.jsonl`.
- It includes an optional `--players-only` refresh path to update player metadata without re-downloading games.

## Data analysis and modelling

Experiments are organized as:

1. `src/1_data_exploration.ipynb` — data quality and leakage-safe feature exploration
2. `src/2_modelling_frequentist.ipynb` — frequentist multiclass baselines
3. `src/2b_modelling_bayes.ipynb` — Bayesian probabilistic check
4. `src/2c_modelling_two_stage.ipynb` — two-stage draw/non-draw, then win/loss decomposition

## Final Recommendation

- **Primary model:** `two_stage_logistic_biased` (`src/2c_modelling_two_stage.ipynb`)
  - best observed **test balanced accuracy** among finalists: **0.5296**
  - best observed **test macro F1** among finalists: **0.5120**
  - non-trivial **test draw recall**: **0.3667**
- **Backup model:** `xgboost_early_stop` (`src/2_modelling_frequentist.ipynb`)
  - simpler one-stage baseline with non-zero draw recall
  - test metrics: balanced accuracy **0.4467**, macro F1 **0.4270**, draw recall **0.3548**

Interpretation:

- The two-stage decomposition gave the strongest class-fair overall result.
- A pure draw-chasing model can raise draw recall further (`softmax_logistic_balanced`: 0.6667) but at a large cost in overall quality (balanced accuracy 0.4552, macro F1 0.3797).

## Most Important Features (and why)

The most useful signals were consistently strength and uncertainty related:

- `rating_delta` (White rating - Black rating): strongest directional signal for win/loss.
- `abs_rating_delta`: closeness proxy; helps separate likely draws from decisive games.
- `rd_delta` and `abs_rd_delta`: rating uncertainty mismatch; captures volatility/instability in expected outcome.
- `stats_win_rate_delta`: form/strength proxy beyond raw rating.
- `stats_draw_rate_delta`: style tendency toward decisive vs drawish games.

Explanation:

- Outcome probability is mostly driven by relative strength (`rating_delta`), while draw probability is more sensitive to **matchup closeness** (`abs_rating_delta`) and **uncertainty/style** (`RD` + draw-rate features).  
- That is exactly why the two-stage approach helped: stage 1 focuses on detecting draw conditions, stage 2 resolves win vs loss once a non-draw is likely.

## Split and Leakage Handling

- only pre-game information is used
- chronological split on March games (`70/15/15`) for train/val/test
- imputation statistics fit on train split only
- stage-2 in two-stage model trained only on non-draw rows

## Evaluation Criteria

Primary:

- `balanced_accuracy`
- `macro_f1`

Guardrail:

- draw recall (`test_draw_recall`)

Secondary:

- multiclass log loss

## Training Objectives by Model

The table below summarizes the optimization objective (loss) used during training for each reported model.

| model | notebook | training objective / loss | class-imbalance handling |
| --- | --- | --- | --- |
| `two_stage_logistic_biased` | `src/2c_modelling_two_stage.ipynb` | Two binary logistic objectives (cross-entropy): Stage 1 `draw` vs `non-draw`, Stage 2 `win` vs `loss` on non-draw subset. Final draw probability adjusted with validation-tuned logit bias. | `class_weight="balanced"` in both logistic stages + validation bias tuning for draw/non-draw threshold behavior |
| `softmax_logistic_balanced` | `src/2c_modelling_two_stage.ipynb` | Multinomial logistic (softmax) negative log-likelihood (cross-entropy) | `class_weight="balanced"` |
| `softmax_rf_balanced` | `src/2c_modelling_two_stage.ipynb` | Random forest split criterion (`gini`) with majority vote / averaged class probabilities | `class_weight="balanced_subsample"` |
| `xgboost_early_stop` | `src/2_modelling_frequentist.ipynb` | XGBoost multiclass objective (`multi:softprob`, i.e., multiclass log-loss) with early stopping on validation set | per-row `sample_weight` from balanced class weights |
| `random_forest_draw_aware` | `src/2_modelling_frequentist.ipynb` | Random forest split criterion (`gini`) + post-hoc draw-threshold tuning on validation set | custom class weights with upweighted draw class (`{0:1, 1:w_draw, 2:1}`) + threshold tuning |
| `bayesian_multinomial_logit` | `src/2b_modelling_bayes.ipynb` | Bayesian multinomial logistic regression with categorical likelihood; trained by posterior inference (NUTS), equivalent likelihood term is multinomial log-loss | imbalance handled through priors/model structure (no explicit `class_weight`) |

### Metric definitions

Notation: $N$ examples, $K=3$ classes (loss, draw, win). Let $y_i \in \{1,\ldots,K\}$ be the true label and $\hat{y}_i$ the predicted class for example $i$. For probabilistic models, let $\hat{p}_{i,c} = P(\hat{y}_i = c \mid x_i)$ with $\sum_c \hat{p}_{i,c} = 1$.

**Confusion counts per class $c$.** Let $TP_c$ be the number of examples with true label $c$ predicted as $c$; $FN_c$ the number with true $c$ predicted as something else; $FP_c$ the number predicted as $c$ but true label not $c$.

**Recall (per-class),** also called sensitivity or hit rate for class $c$:

$$
R_c = \frac{TP_c}{TP_c + FN_c}
$$

**Precision (per-class):**

$$
P_c = \frac{TP_c}{TP_c + FP_c}
$$

**F1 score (per-class):** harmonic mean of precision and recall (with $P_c + R_c = 0$ treated as $F1_c = 0$):

$$
F1_c = \frac{2 P_c R_c}{P_c + R_c}
$$

**Balanced accuracy** (multiclass, as in scikit-learn `balanced_accuracy_score`): the unweighted mean of per-class recalls:

$$
\text{balanced accuracy} = \frac{1}{K} \sum_{c=1}^{K} R_c
$$

**Macro F1:** unweighted mean of per-class F1 scores:

$$
\text{macro-F1} = \frac{1}{K} \sum_{c=1}^{K} F1_c
$$

**Draw recall** (`test_draw_recall`): recall $R_c$ for the class labeled draw (same recall formula with $c = \text{draw}$).

**Multiclass log loss** (cross-entropy; natural log), using class indicator $\mathbb{1}[y_i = c]$:

$$
\text{log loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} \mathbb{1}[y_i = c] \, \log \hat{p}_{i,c}
$$

(Probabilities are clipped to a small $\varepsilon > 0$ where implementations require it to avoid $\log 0$.)

**Overall accuracy** (for context; not the primary selection metric here):

$$
\text{accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]
$$

## Consolidated Final Results


| notebook                            | model                        | test_logloss | test_macro_f1 | test_balanced_accuracy | test_draw_recall | note                                   |
| ----------------------------------- | ---------------------------- | ------------ | ------------- | ---------------------- | ---------------- | -------------------------------------- |
| `src/2c_modelling_two_stage.ipynb`  | `two_stage_logistic_biased`  | 0.9934       | 0.5120        | 0.5296                 | 0.3667           | primary                                |
| `src/2_modelling_frequentist.ipynb` | `xgboost_early_stop`         | 1.0297       | 0.4270        | 0.4467                 | 0.3548           | backup                                 |
| `src/2_modelling_frequentist.ipynb` | `random_forest_draw_aware`   | 0.8902       | 0.4429        | 0.4534                 | 0.0323           | draw-aware RF                          |
| `src/2b_modelling_bayes.ipynb`      | `bayesian_multinomial_logit` | 0.8359       | 0.4440        | 0.4682                 | 0.0000           | better probabilities, weak draw pickup |
| `src/2c_modelling_two_stage.ipynb`  | `softmax_rf_balanced`        | 0.8754       | 0.4407        | 0.4607                 | 0.0000           | one-stage baseline                     |
| `src/2c_modelling_two_stage.ipynb`  | `softmax_logistic_balanced`  | 1.1084       | 0.3797        | 0.4552                 | 0.6667           | high draw recall, weak overall         |


CSV export: `data/processed/final_results_summary.csv`.

Model artifacts and stored outputs:

- final consolidated results are stored in `data/processed/final_results_summary.csv`
- the selected best model artifact is stored in `data/processed/march_best_model.joblib`

## Reproducibility

Run notebooks in order:

1. `src/0_data_preparation.ipynb`
2. `src/1_data_exploration.ipynb`
3. `src/2_modelling_frequentist.ipynb`
4. `src/2b_modelling_bayes.ipynb` (`RUN_BAYES=False` by default)
5. `src/2c_modelling_two_stage.ipynb` (`RUN_BAYES_QUICK=False` by default)

## What I Would Do Next

1. Add stronger draw-specific features:
  - player "theoretical style" proxy (e.g., average ply depth spent in known opening-book lines),
  - opening-structure draw priors (opening family ECO buckets with empirical draw rates),
  - endgame tendency proxies (frequency of simplified material states in recent games),
  - time-management proxies (flagging risk vs controlled time usage from move timestamps when available).
2. Add paired-player interaction statistics (requires more history):
  - head-to-head draw rate for the specific pair,
  - pair-level style compatibility (e.g., tactical-vs-solid matchup indicators),
  - pair-conditioned draw likelihood by time control and opening family.
3. Expand history beyond two events to stabilize rare-class estimates, especially for draw-rate and paired features.
4. Improve calibration and threshold tuning on stricter held-out folds (or rolling-window validation).
5. Add uncertainty-aware decision policy for low-confidence predictions.

