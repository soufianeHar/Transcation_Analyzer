import joblib
from src.data_loader import load_transaction_data
from src.preprocess import get_preprocessor
from src.model import build_model
from src.evaluate import evaluate_model
from src.config import TRAIN_PATH, TEST_PATH, FEATURES, TARGET


# Load training and testing data
df_train = load_transaction_data(TRAIN_PATH, sample_size=20000)
df_test = load_transaction_data(TEST_PATH, sample_size=20000)


# Add "hour" column if not already present
df_train["hour"] = df_train["trans_date_trans_time"].dt.hour
df_test["hour"] = df_test["trans_date_trans_time"].dt.hour

# Prepare inputs and labels
X_train = df_train[FEATURES]
y_train = df_train[TARGET]

X_test = df_test[FEATURES]
y_test = df_test[TARGET]

# Build and train model
preprocessor = get_preprocessor()
model = build_model(preprocessor)
model.fit(X_train, y_train)

# Evaluate on test set
evaluate_model(model, X_test, y_test)

joblib.dump(model, "outputs/fraud_model.joblib")
print("✅ Model saved to outputs/fraud_model.joblib")
