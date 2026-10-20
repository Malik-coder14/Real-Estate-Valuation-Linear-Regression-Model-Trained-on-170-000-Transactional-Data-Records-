# Dubai Residential Property Valuation

A machine-learning project that estimates residential property sale values in Dubai (AED) using **Linear Regression**. The Jupyter notebook explores the transaction data, applies transparent market filters, engineers features, compares models trained on raw and log-transformed sale prices, and exports reusable model artifacts.

## Project objective

Estimate `actual_worth` for Dubai residential **Unit** and **Villa** sales using transaction attributes that could reasonably be known before a sale closes.

This is an educational and analytical valuation estimate—not a formal appraisal or a guarantee of a property's market value.

## Project workflow

1. **Load and profile the data** — inspect dataset dimensions, dates, data types, missing values, duplicate rows, and key categories.
2. **Filter the target market** — retain sales transactions for residential Units and Villas with positive sale values, valid dates, and plausible procedure areas.
3. **Explore the market** — examine price and area distributions and calculate price per square foot.
4. **Engineer features and review leakage** — convert area from square metres to square feet and extract sale year/month. Exclude `meter_sale_price` because it is derived from sale value and area.
5. **Prepare model inputs** — split data into training and test sets (80/20, `random_state=42`); impute missing values, one-hot encode categorical variables, and scale numeric features.
6. **Train and compare models** — compare Linear Regression on the raw sale price and on `log(actual_worth)`. Convert log-model predictions back to AED before evaluating.
7. **Evaluate errors** — calculate R-squared, MAE, RMSE, and MAPE, then inspect errors by community and property type and plot actual versus predicted prices.
8. **Export artifacts** — save the cleaned modelling dataset, metric comparison, and serialized model bundles.

## Features and target

**Target**
- `actual_worth` — transaction sale value in AED.

**Predictor features**
- `area_name_en`
- `property_type_en`
- `property_sub_type_en`
- `rooms_en`
- `reg_type_en`
- `has_parking`
- `size_sqft`
- `sale_year`
- `sale_month`

The notebook intentionally excludes `meter_sale_price`, rent fields, IDs, party-role counts, and duplicate Arabic-language fields to reduce leakage or avoid using fields that are not appropriate predictors.

## Data requirements

The notebook expects a CSV named:

```text
dld_transactions.csv
```

Place it in the same directory as the notebook. The notebook also checks `/mnt/data/files/dld_transactions.csv` as an alternate path. If your data file has a different name or location, update `DATA_PATH` in the setup cell.

The CSV should contain the fields used in the notebook, including transaction date and group, property usage/type/sub-type, `actual_worth`, `procedure_area`, room information, registration type, parking, and area/community.

**Important:** The notebook does not include the raw transaction dataset. Only use data you are authorized to access and share.

## Installation

Recommended: create and activate a virtual environment, then install the packages below.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

Open Jupyter:

```bash
jupyter notebook
```

Open `Dubai_Property_Valuation.ipynb` and run the cells from top to bottom.

## Outputs

After the notebook runs successfully, it creates the folder:

```text
dld_property_valuation_outputs/
```

Expected outputs include:

| File | Description |
|---|---|
| `cleaned_modelling_dataset.csv` | Filtered modelling data containing the selected features and target |
| `model_comparison_metrics.csv` | Test-set comparison of raw-price and log-price Linear Regression |
| `valuation_model_bundle.joblib` | Selected model, preprocessing pipeline, feature list, target, and heuristic percentage range |
| `both_linear_regression_models.joblib` | Both fitted model pipelines and their selected-model metadata |

The selected model is the one with the **lower test-set MAE**. The actual winning model and its metric values depend on the dataset, so consult the notebook outputs rather than assuming a result in advance.

## Evaluation metrics

- **R-squared (R²):** indicates how much variation in sale prices is explained by the model on the test set.
- **MAE (AED):** average absolute prediction error in UAE dirhams.
- **RMSE (AED):** gives larger errors more influence than MAE.
- **MAPE (%):** average absolute error relative to actual sale price, expressed as a percentage.

No single metric fully describes valuation quality. Review the error distribution and results across different communities and property types as well.

## Key assumptions and limitations

- The 10–2,000 m² procedure-area filter is an explicit plausibility assumption and should be reviewed with a local property valuation expert.
- Exact duplicate rows are removed; rows sharing an identifier are not automatically treated as duplicates.
- The random train/test split evaluates held-out records but **does not** establish performance on future transactions. A time-based validation would be useful before operational use.
- The available fields do not fully capture property condition, floor, view, renovation quality, exact unit attributes, negotiation context, or seller motivation.
- Rare or unusually expensive properties may have substantial prediction errors.
- The log-price model's predictions are converted back with `exp()`. This does not guarantee that it will outperform the raw-price model on every metric.
- A simple ±MAPE range is only a heuristic, not a formal prediction interval.
- Human review is recommended, especially for high-value or unusual homes. A production deployment and ongoing monitoring are outside the current notebook.

## Suggested repository structure

```text
Dubai-Property-Valuation/
├── Dubai_Property_Valuation.ipynb
├── dld_transactions.csv                 # Keep private if restricted
├── README.md
└── dld_property_valuation_outputs/      # Generated when notebook runs
```

If the dataset is private, large, or licensed, do **not** commit it to a public GitHub repository. Add it to `.gitignore` and explain how authorized users can obtain it. Avoid committing sensitive transaction data or unnecessary generated artifacts.

## Future improvements

- Validate on later transaction dates using a chronological split.
- Compare with stronger tabular regression models and a simple baseline.
- Report errors by time period, community, property type, and price band.
- Add comparable-sales context and a carefully calibrated uncertainty estimate.
- Build a Streamlit interface only after the selected model and inference pipeline have been validated.

## Disclaimer

This project is for learning and decision support. Predictions are estimates based on the available historical data and selected features; they are not guaranteed sale prices, certified valuations, investment advice, or a substitute for professional assessment.
