import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import normaltest
from sklearn.linear_model import RidgeCV
from statsmodels.stats.stattools import durbin_watson
import time

st.set_page_config(page_title="BLUE Diagnostic System", layout="wide")
st.title("📊 BLUE Assumption Diagnostic System")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file).drop(columns=['Unnamed: 0'])

    df = load_data(uploaded_file)
    st.write("### Data Preview", df.sample(min(500, len(df))))

    target_col = st.selectbox("Select target variable (Y):", df.columns)

    all_features = [col for col in df.columns if col != target_col]
    default_selection = all_features if len(all_features) <= 15 else []

    with st.expander("Select Feature Columns (X)"):
        feature_cols = st.multiselect(
            "Select features for your model:",
            all_features,
            default=default_selection
        )
        if not feature_cols and len(all_features) > 15:
            st.warning("Too many features to auto-select. Please select a subset manually.")

    if not np.issubdtype(df[target_col].dtype, np.number):
        st.warning("🚫 Target column is not numeric. Attempting to convert it...")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    # Detect categorical columns among selected features
    categorical_cols = [col for col in feature_cols if not np.issubdtype(df[col].dtype, np.number)]

    for col in categorical_cols:
        st.warning(f"⚠️ Feature '{col}' is categorical. Please convert this feature to numeric before running diagnostics.")

    if feature_cols and st.button("Run Diagnostics"):
        # Check if any selected features are still categorical
        non_numeric_cols = [col for col in feature_cols if not np.issubdtype(df[col].dtype, np.number)]

        if non_numeric_cols:
            st.warning(f"🚫 Please convert all categorical features to numeric before proceeding: {non_numeric_cols}")
        else:
            y = pd.to_numeric(df[target_col], errors='coerce')
            X = df[feature_cols].apply(pd.to_numeric, errors='coerce')

            missing_before = (X.isna().sum().sum() + y.isna().sum())
            st.write(f"🔍 Missing values before cleaning: {missing_before}")

            data = pd.concat([X, y], axis=1).dropna()

            if data.empty:
                st.error("🚫 Data is empty after removing non-numeric or missing values. Please check your input.")
            else:
                y_clean = data[target_col].values
                X_clean = data[feature_cols].loc[data.index]
                X_const = sm.add_constant(X_clean, has_constant='add')

                model = sm.OLS(y_clean, X_const).fit()
                residuals = model.resid
                fitted = model.fittedvalues

                st.subheader("📈 OLS Summary")
                with st.expander("View Full Summary"):
                    st.text(model.summary())

                sample_size = min(1000, len(fitted))
                sample_idx = np.random.choice(len(fitted), sample_size, replace=False)
                fig, ax = plt.subplots()
                ax.scatter(fitted[sample_idx], residuals[sample_idx], alpha=0.5)
                ax.axhline(0, color='gray', linestyle='dotted')
                if len(fitted[sample_idx]) > 3:
                    z = np.polyfit(fitted[sample_idx], residuals[sample_idx], 3)
                    p = np.poly1d(z)
                    ax.plot(np.sort(fitted[sample_idx]), p(np.sort(fitted[sample_idx])), color='red')
                ax.set_title("Residuals vs Fitted (Linearity)")
                ax.set_xlabel("Fitted values")
                ax.set_ylabel("Residuals")
                st.pyplot(fig)

                # Normality
                stat, p_norm = normaltest(residuals)
                st.subheader("📌 Normality Test")
                st.write(f"D’Agostino Test p-value: {p_norm:.4f}")
                st.write("❌ Residuals are NOT normal" if p_norm < 0.05 else "✅ Residuals appear normal")

                # Heteroscedasticity
                bp_test = sms.het_breuschpagan(residuals, model.model.exog)
                st.subheader("📌 Heteroscedasticity Test")
                st.write(f"Breusch-Pagan p-value: {bp_test[1]:.4f}")
                st.write("❌ Heteroscedasticity detected" if bp_test[1] < 0.05 else "✅ Homoscedasticity holds")

                # Autocorrelation
                dw_stat = durbin_watson(residuals)
                st.subheader("📌 Autocorrelation Test (Durbin-Watson)")
                st.write(f"Durbin-Watson statistic: {dw_stat:.3f}")
                if dw_stat < 1.5 or dw_stat > 2.5:
                    st.write("❌ Possible autocorrelation detected")
                else:
                    st.write("✅ No significant autocorrelation")

                @st.cache_data
                def compute_vif(X_const):
                    return pd.DataFrame({
                        'Feature': X_const.columns,
                        'VIF': [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
                    })

                vif_df = compute_vif(X_const)
                st.subheader("📌 Multicollinearity (VIF Scores)")
                st.dataframe(vif_df)
                if vif_df['VIF'].max() > 5:
                    st.write("❌ Multicollinearity detected")
                else:
                    st.write("✅ No serious multicollinearity")

                # Suggestions
                st.subheader("🧠 Model Suggestions")
                recs = []

                # Linearity (visual only)
                recs.append("Linearity: ⚠️ Check residual plot — transform variables or try nonlinear models if needed")

                # Mean of errors
                if 'const' not in X_const.columns:
                    recs.append("❌ Mean of errors ≠ 0 — add intercept or check model specification")

                # Normality
                if p_norm < 0.05:
                    recs.append("❌ Residuals are not normal — transform Y (log/sqrt) or use robust regression")
                else:
                    recs.append("✅ Residuals appear normal")

                # Heteroscedasticity
                if bp_test[1] < 0.05:
                    recs.append("❌ Heteroscedasticity detected — use Robust SEs, transform Y, or try WLS")
                else:
                    recs.append("✅ Homoscedasticity holds")

                # Autocorrelation
                if dw_stat < 1.5 or dw_stat > 2.5:
                    recs.append("❌ Autocorrelation detected — use GLS, ARIMA, or Newey-West SEs")
                else:
                    recs.append("✅ No serious autocorrelation")

                # Multicollinearity
                if vif_df['VIF'].max() > 5:
                    recs.append("❌ Multicollinearity detected — use Ridge, Lasso, or PCA")
                else:
                    recs.append("✅ No serious multicollinearity")


                for r in recs:
                    st.write("•", r)


else:
    st.info("👆 Upload a CSV file above to begin analysis.")
    
    
    
    
    
