import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score

class RegressionComparison:
    def __init__(self, data_path, target_col, drop_cols=[]):
        self.data = pd.read_csv(data_path).drop(columns=['Unnamed: 0'])
        self.target_col = target_col
        self.drop_cols = drop_cols
        self.results = {}
        
        self.prepare_data()
        
    def prepare_data(self):
        numeric_df = self.data.select_dtypes(include=['number'])
        self.X = numeric_df.drop(columns=self.drop_cols)
        self.y = self.data[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42)
        
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = self.X.columns

    def fit_model(self, model_name, model):
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        r2 = r2_score(self.y_test, y_pred)
        coef = pd.Series(model.coef_, index=self.feature_names)
        
        self.results[model_name] = {
            "model": model,
            "r2_score": r2,
            "coefficients": coef[coef != 0]
        }

    def fit_relaxed_lasso(self, alpha1=0.1):
        # Stage 1: Lasso for feature selection
        lasso = Lasso(alpha=alpha1)
        lasso.fit(self.X_train_scaled, self.y_train)
        mask = lasso.coef_ != 0
        
        # Stage 2: OLS on selected features
        if mask.sum() == 0:
            print("No features selected by Lasso.")
            return
        
        X_train_relaxed = self.X_train_scaled[:, mask]
        X_test_relaxed = self.X_test_scaled[:, mask]
        ols = LinearRegression()
        ols.fit(X_train_relaxed, self.y_train)
        y_pred = ols.predict(X_test_relaxed)
        r2 = r2_score(self.y_test, y_pred)
        
        selected_feature_names = self.feature_names[mask]
        coef = pd.Series(ols.coef_, index=selected_feature_names)
        
        self.results["Relaxed Lasso"] = {
            "model": ols,
            "r2_score": r2,
            "coefficients": coef
        }

    def fit_polynomial_regression(self, degree=2):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(self.X_train_scaled)
        X_test_poly = poly.transform(self.X_test_scaled)

        model = LinearRegression()
        model.fit(X_train_poly, self.y_train)
        y_pred = model.predict(X_test_poly)
        r2 = r2_score(self.y_test, y_pred)

        feature_names = poly.get_feature_names_out(self.feature_names)
        coef = pd.Series(model.coef_, index=feature_names)

        self.results[f"Polynomial (deg={degree})"] = {
            "model": model,
            "r2_score": r2,
            "coefficients": coef[coef != 0]
        }

    def run_all(self):
        self.fit_model("OLS", LinearRegression())
        self.fit_model("Ridge", Ridge(alpha=1.0))
        self.fit_model("Lasso", Lasso(alpha=0.1))
        self.fit_model("Elastic Net", ElasticNet(alpha=0.1, l1_ratio=0.5))
        self.fit_relaxed_lasso(alpha1=0.1)
        self.fit_polynomial_regression(degree=2)

    def summary(self):
        print("\n📊 Model Performance (R² Score):")
        for name, result in self.results.items():
            print(f"{name}: R² = {result['r2_score']:.4f}, Features used = {len(result['coefficients'])}")
        
        return self.results


# ========== Run ==========
regressor = RegressionComparison(
    data_path="/home/ccp/Desktop/LR/archive/data/df_features.csv",
    target_col='log_funding_total_usd',
    drop_cols=['log_funding_total_usd', 'funding_total_usd']
)

regressor.run_all()
results = regressor.summary()
