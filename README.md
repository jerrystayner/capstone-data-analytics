# Mesa Obesity Capstone Project

Final project for CIS 480. Looks at how income, education, insurance coverage, commuting, and grocery store access relate to adult obesity rates across census tracts in Mesa, Arizona.

## Overview

I pulled tract-level data from PolicyMap and grocery store locations from the USDA SNAP Retailer Locator, cleaned and merged everything by FIPS code, then I ran a multiple linear regression to see which factors actually predict obesity rates.

The model:

```
obesity_pct ~ median_income + grocery_per_10k + drove_pct + bachelors_pct + uninsured_pct
```

Results, Power Bi Dashboard, and charts are in the visualizations folder. The complete write-up is in documentation.

## Technologies Used

- Python 3 (pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn, scipy)
- Microsoft Power BI
- Data from PolicyMap and the USDA SNAP Retailer Locator

## How to Reproduce

1. Clone the repo.
2. Install the Python packages:

   ```
   pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy
   ```

3. From the project folder, run the scripts in order:

   ```
   python scripts/01_clean_and_merge.py
   python scripts/02_compute_grocery_density.py
   python scripts/03_exploratory_analysis.py
   python scripts/04_regression_model.py
   ```

4. Open `visualizations/Mesa_Obesity_Dashboard.pbix` in Power BI Desktop to view the dashboard.

## File Structure

```
capstone-data-analytics/
├── README.md
├── datasets/         raw CSVs and merged data
├── scripts/          Python scripts (run in order 01 -> 04)
├── visualizations/   charts, regression outputs, and the .pbix dashboard
└── documentation/    final report and Power BI instructional report
```

## Author

Jerry Stayner, CIS 480
