from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def build_model(preprocessor):
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=20,
            random_state=42
        ))
    ])
    return model
