# ==============================================================================
# Diabetes Risk Prediction — Pima Indians Diabetes Dataset
# Comparing Logistic Regression, Decision Tree, Random Forest, and XGBoost
#
# Dataset : 768 rows. Columns: pregnancies, glucose, blood_pressure,
#           skin_thickness, insulin, bmi, diabetes_pedigree_function, age,
#           outcome ("diabetic" / "Nondiabetic")
# Author  : <your name>
# Updated : 2026
#
# Key data quality issue this script addresses:
#   Glucose, BloodPressure, SkinThickness, Insulin, and BMI contain literal
#   zeros that are physiologically impossible (blood pressure of 0, BMI of
#   0.0) — these are missing measurements recorded as 0, not real readings.
#   Left untreated, they silently bias every model trained on them. This
#   script converts them to NA and imputes rather than trusting them as-is.
# ==============================================================================


# ---- 1. Setup ----------------------------------------------------------------
# renv::restore()  # uncomment if this project is locked with {renv}

required_pkgs <- c(
  "tidyverse",    # dplyr, ggplot2, tidyr, readr, purrr
  "here",         # OS-independent, project-relative file paths
  "skimr",        # richer alternative to summary()
  "janitor",      # clean_names() for consistent snake_case columns
  "caret",        # train/test split, resampling, unified model interface
  "rpart",        # decision tree
  "rpart.plot",   # plotting rpart trees
  "randomForest", # random forest engine used via caret
  "xgboost",      # gradient boosted trees
  "pROC",         # ROC curves & AUC
  "ggcorrplot",   # correlation heatmap
  "e1071"         # skewness() diagnostic, also required internally by caret for some models
)

new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs) > 0) install.packages(new_pkgs)
invisible(lapply(required_pkgs, library, character.only = TRUE))

options(scipen = 999)
set.seed(101)


# ---- 2. Load data --------------------------------------------------------------
data_path <- here("data", "diabetes.csv")

data_raw <- read_csv(data_path, show_col_types = FALSE) %>%
  clean_names()  # -> pregnancies, glucose, blood_pressure, skin_thickness,
                 #    insulin, bmi, diabetes_pedigree_function, age, outcome

glimpse(data_raw)
skim(data_raw)

table(data_raw$outcome, useNA = "always")  # confirm labels before recoding anything


# ---- 3. Clean & recode ------------------------------------------------------

# --- Outcome ---
# Handles text labels robustly ("diabetic"/"Nondiabetic" here) or numeric
# 0/1 if your copy of the dataset uses that encoding instead — check either
# way with the table() call above before trusting this.
data <- data_raw %>%
  mutate(
    outcome = case_when(
      tolower(as.character(outcome)) %in% c("1", "diabetic", "pos", "yes") ~ "Diabetic",
      tolower(as.character(outcome)) %in% c("0", "nondiabetic", "neg", "no") ~ "NonDiabetic",
      TRUE ~ NA_character_
    ),
    outcome = factor(outcome, levels = c("NonDiabetic", "Diabetic"))
  )

table(data$outcome, useNA = "always")
# Drop any row where the outcome couldn't be matched — a row with no usable
# label is not something a supervised model can learn from or be scored on.
data <- data %>% filter(!is.na(outcome))

# --- Disguised missing values ---
# Zero is not a valid reading for these five columns; convert to NA so it
# can be imputed properly instead of treated as a real measurement of zero.
zero_as_na_cols <- c("glucose", "blood_pressure", "skin_thickness", "insulin", "bmi")
data <- data %>%
  mutate(across(all_of(zero_as_na_cols), ~ na_if(., 0)))

cat("Missing values introduced per column:\n")
print(colSums(is.na(data)))
# Insulin and SkinThickness are typically the most affected (often ~30-50%
# missing) — this is a known, well-documented property of this dataset, not
# a bug in this script. Median imputation (in the modeling pipeline below)
# is the standard, defensible approach; more advanced options like KNN or
# MICE imputation exist if you want to push this further for a portfolio.

# --- Clinical range validation (flag, don't blindly trust the raw values) ---
range_flags <- data %>%
  summarise(
    pregnancies_out = sum(pregnancies < 0 | pregnancies > 20, na.rm = TRUE),
    glucose_out      = sum(glucose < 40 | glucose > 300, na.rm = TRUE),
    bp_out           = sum(blood_pressure < 20 | blood_pressure > 200, na.rm = TRUE),
    bmi_out          = sum(bmi < 10 | bmi > 80, na.rm = TRUE),
    age_out          = sum(age < 0 | age > 120, na.rm = TRUE)
  )
print(range_flags)  # investigate/remove any flagged rows if counts are nonzero

# --- Contradictory records ---
# Identical predictor values with two different outcome labels give the
# model conflicting signal to learn from. Remove them rather than let them
# add noise (rare in this dataset, but worth checking rather than assuming).
predictor_cols <- setdiff(names(data), "outcome")
contradictions <- data %>%
  group_by(across(all_of(predictor_cols))) %>%
  filter(n_distinct(outcome) > 1) %>%
  ungroup()
cat("Contradictory rows (identical predictors, different outcome):", nrow(contradictions), "\n")

data <- data %>%
  group_by(across(all_of(predictor_cols))) %>%
  filter(n_distinct(outcome) == 1) %>%
  ungroup()

# --- Exact duplicates ---
n_dupes <- sum(duplicated(data))
cat("Exact duplicate rows removed:", n_dupes, "\n")
data <- data %>% distinct()

# --- Skewness check on numeric predictors ---
# Relevant mainly for the logistic regression's linearity assumption;
# tree-based models (rf, xgboost) are unaffected by skew or scale.
data %>%
  summarise(across(where(is.numeric), ~ round(skewness(.x, na.rm = TRUE), 2))) %>%
  print()

cat("Final row count after cleaning:", nrow(data), "\n")


# ---- 4. Exploratory data analysis --------------------------------------------
table(data$outcome)
prop.table(table(data$outcome))  # ~65% non-diabetic / 35% diabetic — moderate imbalance

# Distribution of each numeric predictor
data %>%
  select(-outcome) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 30, fill = "steelblue", na.rm = TRUE) +
  facet_wrap(~variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of predictors (post-cleaning)", x = NULL, y = "Count")

# Boxplots by outcome
data %>%
  pivot_longer(-outcome, names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = outcome, y = value, fill = outcome)) +
  geom_boxplot(na.rm = TRUE, outlier.alpha = 0.3) +
  facet_wrap(~variable, scales = "free_y") +
  theme_minimal() +
  labs(title = "Predictor distributions by diabetes outcome") +
  theme(legend.position = "none")

# Correlation heatmap
corr_matrix <- data %>%
  select(where(is.numeric)) %>%
  cor(use = "pairwise.complete.obs")

ggcorrplot(corr_matrix, lab = TRUE, lab_size = 3, type = "lower",
           title = "Correlation between predictors")


# ---- 5. Train / test split ---------------------------------------------------
set.seed(101)
train_idx  <- createDataPartition(data$outcome, p = 0.7, list = FALSE)
train_data <- data[train_idx, ]
test_data  <- data[-train_idx, ]

table(train_data$outcome)
table(test_data$outcome)


# ---- 6. Cross-validation setup & class-imbalance handling --------------------
# Sampling is applied inside each CV fold (not once on the whole training
# set up front) so the held-out fold in every resample stays untouched by
# resampling — this avoids an overly optimistic performance estimate.
# "up" is used here (not "down" as in the larger dataset) because this
# dataset is small; down-sampling a already-small training set would throw
# away too much of the majority class to leave much to learn from.
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "up",
  savePredictions = "final"
)

# medianImpute fills the NAs introduced for the disguised-zero columns;
# center/scale standardizes numeric predictors for glm convergence and
# consistency across model types.
pre_proc <- c("medianImpute", "center", "scale")


# ---- 7. Model training --------------------------------------------------------
# na.action = na.pass on every call: without it, caret's default (na.fail)
# stops training the moment it sees any NA, before preProcess ever runs.
set.seed(101)
model_glm <- train(
  outcome ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  metric = "ROC",
  trControl = ctrl,
  preProcess = pre_proc,
  na.action = na.pass
)

set.seed(101)
model_rpart <- train(
  outcome ~ .,
  data = train_data,
  method = "rpart",
  metric = "ROC",
  trControl = ctrl,
  preProcess = pre_proc,
  na.action = na.pass,
  tuneLength = 10
)
rpart.plot(model_rpart$finalModel, main = "Decision tree (best CV tune)")

set.seed(101)
model_rf <- train(
  outcome ~ .,
  data = train_data,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  preProcess = pre_proc,
  na.action = na.pass,
  tuneLength = 5,
  importance = TRUE
)

set.seed(101)
model_xgb <- train(
  outcome ~ .,
  data = train_data,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl,
  preProcess = pre_proc,
  na.action = na.pass,
  tuneLength = 5,
  verbosity = 0
)


# ---- 8. Compare models on cross-validated performance -------------------------
resamps <- resamples(list(
  Logistic     = model_glm,
  DecisionTree = model_rpart,
  RandomForest = model_rf,
  XGBoost      = model_xgb
))

summary(resamps)
bwplot(resamps, metric = "ROC")
dotplot(resamps)


# ---- 9. Hold-out test set evaluation -------------------------------------------
evaluate_model <- function(model, test_data, model_name) {
  probs <- predict(model, newdata = test_data, type = "prob")[, "Diabetic"]
  preds <- predict(model, newdata = test_data)
  cm <- confusionMatrix(preds, test_data$outcome, positive = "Diabetic")
  roc_obj <- roc(response = test_data$outcome, predictor = probs,
                  levels = rev(levels(test_data$outcome)), quiet = TRUE)

  list(
    model = model_name,
    accuracy = unname(cm$overall["Accuracy"]),
    sensitivity = unname(cm$byClass["Sensitivity"]),
    specificity = unname(cm$byClass["Specificity"]),
    precision = unname(cm$byClass["Precision"]),
    f1 = unname(cm$byClass["F1"]),
    auc = as.numeric(auc(roc_obj)),
    roc_obj = roc_obj,
    confusion_matrix = cm
  )
}

results <- list(
  Logistic     = evaluate_model(model_glm,   test_data, "Logistic Regression"),
  DecisionTree = evaluate_model(model_rpart, test_data, "Decision Tree"),
  RandomForest = evaluate_model(model_rf,    test_data, "Random Forest"),
  XGBoost      = evaluate_model(model_xgb,   test_data, "XGBoost")
)

test_summary <- purrr::map_dfr(results, ~ tibble(
  Model = .x$model,
  Accuracy = .x$accuracy,
  Sensitivity = .x$sensitivity,
  Specificity = .x$specificity,
  Precision = .x$precision,
  F1 = .x$f1,
  AUC = .x$auc
))

print(test_summary)
results$RandomForest$confusion_matrix

plot(results$Logistic$roc_obj, col = "steelblue", main = "ROC curves - test set")
plot(results$DecisionTree$roc_obj, col = "forestgreen", add = TRUE)
plot(results$RandomForest$roc_obj, col = "darkorange", add = TRUE)
plot(results$XGBoost$roc_obj, col = "purple", add = TRUE)
legend("bottomright",
       legend = paste0(test_summary$Model, " (AUC = ", round(test_summary$AUC, 3), ")"),
       col = c("steelblue", "forestgreen", "darkorange", "purple"), lwd = 2, cex = 0.8)


# ---- 10. Feature importance for the best model -----------------------------------
best_model_name <- test_summary$Model[which.max(test_summary$AUC)]
cat("Best model on the held-out test set:", best_model_name, "\n")

plot(varImp(model_rf), main = "Random forest variable importance")


# ---- 11. Save artifacts for reuse / reproducibility ------------------------------
dir.create(here("models"), showWarnings = FALSE)
saveRDS(model_rf, here("models", "random_forest_diabetes.rds"))
writeLines(capture.output(sessionInfo()), here("sessionInfo.txt"))
