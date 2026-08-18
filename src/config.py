DATA_PATH = "data/dataset.csv"
MODEL_DIR = "saved_models"

TARGET_COL = "price"
NUMERICAL_FEATURES = ["cost", "competitor_price", "demand_score", "seasonality", "rating", "marketing_spend"]
CATEGORICAL_FEATURES = []
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
