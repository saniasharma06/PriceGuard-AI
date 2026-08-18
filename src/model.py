import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import root_mean_squared_error, accuracy_score, classification_report
from src.utils import ensure_dir

class RegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def evaluate(self, X, y):
        preds = self.predict(X)
        rmse = root_mean_squared_error(y, preds)
        return rmse

class ClassificationModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def evaluate(self, X, y):
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        report = classification_report(y, preds, output_dict=True)
        return acc, report

class AnomalyModel:
    def __init__(self):
        # contamination=0.05 matches our generated 5% anomalies
        self.model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        
    def train(self, X):
        self.model.fit(X)
        
    def predict(self, X):
        # Isolation Forest returns 1 for inliers and -1 for outliers
        preds = self.model.predict(X)
        return preds

def save_model(model, path):
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)
    
def load_model(path):
    return joblib.load(path)
