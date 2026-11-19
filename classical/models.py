"""Classical models: logistic regression, random forest and XGBoost on the 23 similarity features
(helpers/features.py), each with its library defaults (random_state=0 for the two tree models).

Features: helpers.features.get_processed_dfs(..., missing_number_as_zero=False) gives the sentinel setting
(the paper's default), missing_number_as_zero=True the zero-fill ablation.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# Fit each classifier on train features, predict on val and test features.
# Returns {model_name: {"val": [preds, probs], "test": [preds, probs]}}.
def _predict_val_test(model, val_df, test_df):
    return {
        split: [model.predict(X), model.predict_proba(X)[:, 1]]
        for split, X in [("val", val_df), ("test", test_df)]
    }


# return_models=True also returns the fitted models: (ret, {model_name: fitted model}), e.g. for feature importances.
def run_against_models(train_df, train_label, val_df, test_df, return_models=False):
    ret = {}
    models = {}

    model = LogisticRegression()
    model.fit(train_df, train_label)
    ret["LogisticRegression"] = _predict_val_test(model, val_df, test_df)
    models["LogisticRegression"] = model

    rf = RandomForestClassifier(random_state=0)    # n_estimators=100 by default
    rf.fit(train_df, train_label)
    ret["RandomForestClassifier"] = _predict_val_test(rf, val_df, test_df)
    models["RandomForestClassifier"] = rf

    xgb = XGBClassifier(random_state=0)
    xgb.fit(train_df, train_label)
    ret["XGBClassifier"] = _predict_val_test(xgb, val_df, test_df)
    models["XGBClassifier"] = xgb

    return (ret, models) if return_models else ret
