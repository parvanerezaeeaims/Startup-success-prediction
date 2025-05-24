# Startup Funding Analysis

This repository contains data exploration and regression analysis on startup funding data. The goal is to understand what factors influence funding success, using statistical methods and visualizations.

## 📊 Project Objectives

- Explore the relationship between funding and startup characteristics.
- Apply Linear Regression to predict total funding.
- Evaluate model validity using **BLUE** assumptions (Best Linear Unbiased Estimator).
- Visualize funding vs. success (e.g., acquisition, IPO).
- Handle outliers and apply data transformations (log, normalization).

## 📁 Contents

- `data`: The dataset used for analysis.
- `notebooks/`: Jupyter notebooks for preprocessing, modeling, and diagnostics.
- `images/`: Key visualizations like violin plots, residual plots, and Q-Q plots.


## ⚙️ Key Techniques

- **Linear Regression** with `statsmodels` and `scikit-learn`
- **Outlier detection and treatment**
- **Log transformation** and **scaling**
- **Assumption checks**:
  - Linearity
  - Normality of residuals (Shapiro-Wilk, Q-Q plot)
  - Homoscedasticity (Breusch-Pagan test)
  - No autocorrelation (Durbin-Watson)
  - No multicollinearity (VIF)
- **Interactive visualizations** with `plotly`

## 📈 Example Visualization

```python
# Violin plot with log-scaled y-axis
fig = px.violin(df, x='status', y='funding_total_usd', color='status',
                box=True, points='all',
                title='Funding Distribution by Startup Status',
                labels={'funding_total_usd': 'Total Funding (USD)', 'status': 'Startup Status'})
fig.update_yaxes(type='log')
fig.show()
```

## 📬 Contact

For feedback or collaboration:[parvanerezaeeaims@gmail.com]

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork the repository, open issues, suggest new ideas, or submit a pull request.  
Whether it's improving documentation, fixing bugs, or enhancing the analysis—every bit helps! 🚀

