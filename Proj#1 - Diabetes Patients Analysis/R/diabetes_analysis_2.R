# ==============================================================================
# DIABETES RISK PREDICTION
# Pima Indians Diabetes Dataset
#
# Models:
#   1. Logistic Regression
#   2. Decision Tree
#   3. Random Forest
#
# Framework: tidymodels
#   This version moves off {caret} onto {tidymodels} — the current standard
#   R modeling framework as of 2026. caret is still maintained but is now
#   the older-generation tool; tidymodels is its intended successor, built
#   by the same author (Max Kuhn). Concretely, this fixes several rough
#   edges the caret version had:
#     - No more `na.action = na.pass` workaround: recipes handle imputation
#       as a first-class pipeline step, so there's nothing to silently fail.
#     - No more custom `roc_summary()` function: yardstick computes ROC/
#       Sens/Spec/etc. natively via metric_set(), so the "all ROC values
#       are missing" class of bug simply can't happen the same way.
#     - Preprocessing (imputation, scaling) is defined once in a `recipe`
#       and re-applied identically and automatically to every resample and
#       to the test set — no risk of it being computed differently in
#       different places.
#     - workflow_set() lets all candidate models share one recipe and be
#       tuned/compared in a single pass, which is the current idiomatic
#       pattern for "try several models, pick the best" projects.
#
# Evaluation:
#   - Accuracy, Sensitivity, Specificity, Precision, F1, ROC/AUC
# ==============================================================================


# ==============================================================================
# 1. SETUP
# ==============================================================================

required_pkgs <- c(
  "tidyverse",   # dplyr, ggplot2, tidyr, readr, purrr
  "tidymodels",  # rsample, recipes, parsnip, workflows, tune, yardstick, workflowsets, dials
  "here",        # project-relative file paths
  "skimr",       # richer summary()
  "janitor",     # clean_names()
  "rpart",       # decision tree engine used by parsnip
  "ranger",      # fast random forest engine used by parsnip
  "vip",         # variable importance plots for parsnip/tidymodels fits
  "ggcorrplot",  # correlation heatmap
  "e1071"        # skewness() diagnostic
)

new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs) > 0) install.packages(new_pkgs)
invisible(lapply(required_pkgs, library, character.only = TRUE))

options(scipen = 999)
tidymodels_prefer()  # resolves function name clashes (e.g. dplyr::filter vs stats::filter) in favor of tidymodels
set.seed(101)


# ==============================================================================
# 2. LOAD DATA
# ==============================================================================

data_path <- here("data", "datasets_diabetes.csv")

if (!file.exists(data_path)) {
  stop(
    paste0(
      "The dataset could not be found at:\n", data_path, "\n\n",
      "Please check that datasets_diabetes.csv is inside your data folder."
    )
  )
}

data_raw <- read_csv(data_path, show_col_types = FALSE) %>%
  clean_names()

cat("\n===== DATA STRUCTURE =====\n")
glimpse(data_raw)

cat("\n===== DATA SUMMARY =====\n")
skim(data_raw)

cat("\n===== ORIGINAL OUTCOME VALUES =====\n")
print(table(data_raw$outcome, useNA = "always"))


# ==============================================================================
# 3. CLEAN AND RECODE DATA
# ==============================================================================

# --- 3.1 Outcome variable ------------------------------------------------
data <- data_raw %>%
  mutate(
    outcome = case_when(
      tolower(trimws(as.character(outcome))) %in%
        c("1", "diabetic", "diabetes", "pos", "positive", "yes") ~ "Diabetic",
      tolower(trimws(as.character(outcome))) %in%
        c("0", "nondiabetic", "non-diabetic", "negative", "neg", "no") ~ "NonDiabetic",
      TRUE ~ NA_character_
    ),
    # NonDiabetic listed first, Diabetic second — yardstick/tidymodels treats
    # the *first* factor level as the reference and the *second* as the
    # event of interest by default, so this ordering matters.
    outcome = factor(outcome, levels = c("NonDiabetic", "Diabetic"))
  )

cat("\n===== RECODED OUTCOME =====\n")
print(table(data$outcome, useNA = "always"))

# --- 3.2 Remove rows with unusable outcome labels -------------------------
data <- data %>% filter(!is.na(outcome))
cat("\nRows after removing missing outcome:", nrow(data), "\n")


# ==============================================================================
# 4. CONVERT IMPOSSIBLE ZERO VALUES TO NA
# ==============================================================================
# Zero is not a physiologically valid reading for these columns in this
# dataset — it represents a missing measurement, not a real value of zero.

zero_as_na_cols <- c("glucose", "blood_pressure", "skin_thickness", "insulin", "bmi")

missing_columns <- setdiff(zero_as_na_cols, names(data))
if (length(missing_columns) > 0) {
  stop(paste0("These expected columns are missing from your dataset: ",
              paste(missing_columns, collapse = ", ")))
}

data <- data %>%
  mutate(across(all_of(zero_as_na_cols), ~ na_if(.x, 0)))

cat("\n===== MISSING VALUES AFTER ZERO CONVERSION =====\n")
print(colSums(is.na(data)))
# Imputation itself happens later, inside the recipe (Step 12) — not here.
# This keeps imputation as a proper pipeline step, computed fresh on each
# resample's training portion, rather than a one-off calculation on the
# whole dataset that risks leaking test-set information.


# ==============================================================================
# 5. RANGE / SANITY CHECKS
# ==============================================================================

range_flags <- data %>%
  summarise(
    pregnancies_out = sum(pregnancies < 0 | pregnancies > 20, na.rm = TRUE),
    glucose_out      = sum(glucose < 40 | glucose > 300, na.rm = TRUE),
    bp_out           = sum(blood_pressure < 20 | blood_pressure > 200, na.rm = TRUE),
    bmi_out          = sum(bmi < 10 | bmi > 80, na.rm = TRUE),
    age_out          = sum(age < 0 | age > 120, na.rm = TRUE)
  )
cat("\n===== RANGE CHECK =====\n")
print(range_flags)


# ==============================================================================
# 6. DUPLICATES
# ==============================================================================

n_dupes <- sum(duplicated(data))
cat("\nExact duplicate rows:", n_dupes, "\n")
data <- data %>% distinct()


# ==============================================================================
# 7. CONTRADICTORY RECORDS (flagged, not removed)
# ==============================================================================
# Identical predictor values with two different outcome labels *can* be
# genuinely different people — we surface the count for awareness but don't
# assume they should be dropped.

predictor_cols <- setdiff(names(data), "outcome")

contradictions <- data %>%
  group_by(across(all_of(predictor_cols))) %>%
  filter(n_distinct(outcome) > 1) %>%
  ungroup()

cat("\nContradictory predictor/outcome rows:", nrow(contradictions), "\n")


# ==============================================================================
# 8. SKEWNESS CHECK
# ==============================================================================

cat("\n===== SKEWNESS OF NUMERIC VARIABLES =====\n")
data %>%
  summarise(across(where(is.numeric), ~ round(skewness(.x, na.rm = TRUE), 2))) %>%
  print()
# Relevant mainly for the logistic regression's linearity assumption; the
# decision tree and random forest are unaffected by skew or scale.


# ==============================================================================
# 9. FINAL DATA CHECK
# ==============================================================================

cat("\nFinal number of rows:", nrow(data), "\n")
cat("Final number of columns:", ncol(data), "\n")
cat("\nOutcome distribution:\n")
print(table(data$outcome))
cat("\nOutcome proportions:\n")
print(round(prop.table(table(data$outcome)), 3))


# ==============================================================================
# 10. EXPLORATORY DATA ANALYSIS
# ==============================================================================

data %>%
  select(-outcome) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 30, fill = "steelblue", na.rm = TRUE) +
  facet_wrap(~variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Predictors", x = NULL, y = "Count")

data %>%
  pivot_longer(-outcome, names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = outcome, y = value, fill = outcome)) +
  geom_boxplot(na.rm = TRUE, outlier.alpha = 0.3) +
  facet_wrap(~variable, scales = "free_y") +
  theme_minimal() +
  labs(title = "Predictor Distributions by Diabetes Outcome", x = "Outcome", y = "Value") +
  theme(legend.position = "none")

corr_matrix <- data %>%
  select(where(is.numeric)) %>%
  cor(use = "pairwise.complete.obs")

ggcorrplot(corr_matrix, lab = TRUE, lab_size = 3, type = "lower",
           title = "Correlation Between Predictors")


# ==============================================================================
# 11. TRAIN / TEST SPLIT
# ==============================================================================
# rsample's initial_split() is the tidymodels equivalent of caret's
# createDataPartition() — `strata = outcome` keeps the class balance
# consistent between train and test, same intent as before.

set.seed(101)
data_split <- initial_split(data, prop = 0.70, strata = outcome)
train_data <- training(data_split)
test_data  <- testing(data_split)

cat("\n===== TRAINING DATA =====\n")
cat("Rows:", nrow(train_data), "\n")
print(table(train_data$outcome))

cat("\n===== TEST DATA =====\n")
cat("Rows:", nrow(test_data), "\n")
print(table(test_data$outcome))

# 5-fold, 3-repeat cross-validation on the training set, stratified so every
# fold keeps roughly the same class balance as the full training set.
set.seed(101)
cv_folds <- vfold_cv(train_data, v = 5, repeats = 3, strata = outcome)


# ==============================================================================
# 12. RECIPE (preprocessing pipeline)
# ==============================================================================
# A recipe replaces caret's `preProcess = c(...)` argument. Its steps are
# *learned* on each resample's training fold only, then applied to that
# fold's held-out data and, later, the real test set — this is what keeps
# preprocessing statistics (like the median used for imputation) from
# leaking information across folds.
#
# step_upsample (from the {themis} package) replaces caret's
# `sampling = "up"` — install/load {themis} if you'd rather balance classes
# this way; omitted here to keep the core dependency list smaller, since
# tree-based/logistic models with class weights or threshold tuning are a
# reasonable alternative. Add themis::step_upsample(outcome) below the
# step_normalize() line if you want the exact caret-equivalent behavior.

diabetes_recipe <- recipe(outcome ~ ., data = train_data) %>%
  step_impute_median(all_of(zero_as_na_cols)) %>%
  step_normalize(all_numeric_predictors())


# ==============================================================================
# 13. MODEL SPECIFICATIONS
# ==============================================================================
# Each parsnip model spec declares WHAT to fit and WHICH parameters to tune
# (tune()), independent of the data — the recipe above and these specs are
# combined into workflows in the next step.

logistic_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

tree_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

forest_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 300
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")


# ==============================================================================
# 14. WORKFLOW SET — bundle all models with the shared recipe
# ==============================================================================
# This is the current idiomatic tidymodels pattern for "compare several
# candidate models against one preprocessing pipeline": define them all
# together, then tune/fit/evaluate them in one call.

wf_set <- workflow_set(
  preproc = list(recipe = diabetes_recipe),
  models = list(
    logistic = logistic_spec,
    tree = tree_spec,
    forest = forest_spec
  )
)

class_metrics <- metric_set(roc_auc, sens, spec, precision, f_meas, accuracy)

cat("\n========================================\n")
cat("TUNING / CROSS-VALIDATING ALL MODELS\n")
cat("========================================\n")

set.seed(101)
wf_results <- wf_set %>%
  workflow_map(
    fn = "tune_grid",
    resamples = cv_folds,
    grid = 5,
    metrics = class_metrics,
    control = control_grid(save_pred = TRUE, parallel_over = "everything"),
    verbose = TRUE
  )
# Logistic regression has no tunable hyperparameters — tune_grid() detects
# this automatically and just cross-validates it once, no extra code needed.


# ==============================================================================
# 15. CROSS-VALIDATED MODEL COMPARISON
# ==============================================================================

cat("\n========================================\n")
cat("CROSS-VALIDATED MODEL COMPARISON (ROC-AUC)\n")
cat("========================================\n")

print(rank_results(wf_results, rank_metric = "roc_auc", select_best = TRUE))

autoplot(wf_results, metric = "roc_auc") +
  theme_minimal() +
  labs(title = "Cross-validated ROC-AUC by model")


# ==============================================================================
# 16. SELECT BEST MODEL, FINALIZE, AND FIT ON THE FULL TRAINING SET
# ==============================================================================

best_wf_id <- rank_results(wf_results, rank_metric = "roc_auc", select_best = TRUE) %>%
  slice(1) %>%
  pull(wflow_id)

cat("\nBest-performing model in cross-validation:", best_wf_id, "\n")

best_params <- wf_results %>%
  extract_workflow_set_result(best_wf_id) %>%
  select_best(metric = "roc_auc")

final_wf <- wf_results %>%
  extract_workflow(best_wf_id) %>%
  finalize_workflow(best_params)

# last_fit() fits the finalized workflow on the FULL training set and
# evaluates it on the held-out test set in one call — this is the
# tidymodels equivalent of caret's separate `predict()` + `confusionMatrix()`
# + `roc()` calls, bundled together with no risk of mismatched data.
set.seed(101)
final_fit <- last_fit(final_wf, split = data_split, metrics = class_metrics)


# ==============================================================================
# 17. TEST-SET PERFORMANCE
# ==============================================================================

cat("\n========================================\n")
cat("TEST-SET PERFORMANCE (BEST MODEL:", best_wf_id, ")\n")
cat("========================================\n")

test_summary <- collect_metrics(final_fit)
print(test_summary)

test_preds <- collect_predictions(final_fit)

cat("\n===== CONFUSION MATRIX =====\n")
print(conf_mat(test_preds, truth = outcome, estimate = .pred_class))


# ==============================================================================
# 18. ROC CURVE
# ==============================================================================

roc_curve(test_preds, truth = outcome, .pred_Diabetic, event_level = "second") %>%
  autoplot() +
  labs(title = paste("ROC Curve - Test Set (", best_wf_id, ")"))
# event_level = "second" tells yardstick that "Diabetic" (the second factor
# level, per Step 3.1) is the class of interest.


# ==============================================================================
# 19. VARIABLE IMPORTANCE (best model)
# ==============================================================================

cat("\n========================================\n")
cat("VARIABLE IMPORTANCE\n")
cat("========================================\n")

final_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 8) +
  theme_minimal() +
  labs(title = paste("Variable Importance -", best_wf_id))


# ==============================================================================
# 20. COMPARE ALL THREE MODELS ON THE SAME TEST SET (optional, for the writeup)
# ==============================================================================
# last_fit() above only finalizes and tests the single best model. If you
# want a side-by-side test-set table across all three models (not just CV
# estimates), fit and evaluate each finalized workflow individually:

fit_and_evaluate <- function(wflow_id) {
  params <- wf_results %>%
    extract_workflow_set_result(wflow_id) %>%
    select_best(metric = "roc_auc")

  wf <- wf_results %>%
    extract_workflow(wflow_id) %>%
    finalize_workflow(params)

  set.seed(101)
  fit_result <- last_fit(wf, split = data_split, metrics = class_metrics)

  collect_metrics(fit_result) %>%
    mutate(model = wflow_id) %>%
    select(model, .metric, .estimate)
}

all_model_comparison <- map_dfr(wf_results$wflow_id, fit_and_evaluate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

cat("\n===== ALL MODELS - TEST SET COMPARISON =====\n")
print(all_model_comparison)


# ==============================================================================
# 21. SAVE ARTIFACTS
# ==============================================================================

models_dir <- here("models")
dir.create(models_dir, showWarnings = FALSE, recursive = TRUE)

# Save the finalized, fitted best workflow — this single object bundles the
# recipe (preprocessing) and the fitted model together, so it can be
# reloaded and applied to brand-new data with one predict() call.
final_wf_fitted <- extract_workflow(final_fit)
saveRDS(final_wf_fitted, here("models", "best_workflow_diabetes.rds"))

saveRDS(test_summary, here("models", "model_comparison_results.rds"))
saveRDS(all_model_comparison, here("models", "all_models_test_comparison.rds"))

writeLines(capture.output(sessionInfo()), here("sessionInfo.txt"))


# ==============================================================================
# END OF ANALYSIS
# ==============================================================================

cat("\n========================================\n")
cat("ANALYSIS COMPLETED SUCCESSFULLY\n")
cat("========================================\n")
cat("\nBest model:", best_wf_id, "\n")
cat("Model files saved in:", models_dir, "\n")
