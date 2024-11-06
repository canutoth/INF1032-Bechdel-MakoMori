import pandas as pd
import statsmodels.api as sm
import numpy as np
import json
from scipy.stats import spearmanr

# Load JSON data
with open('C:/Users/AISE LAB/Documents/inf1032/INF1032-Bechdel-MakoMori/data/summedUp_data.json', 'r') as file:
    data = json.load(file)

# Create a DataFrame from JSON data
df = pd.DataFrame(data)

# Define criteria for passing the tests
df['bechdel_pass'] = df['bechdel'] >= 3
df['mako_mori_pass'] = df['mako-mori'] > 0

# Convert test results to binary variables for regression analysis
df['Bechdel'] = df['bechdel_pass'].astype(int)
df['Mako_Mori'] = df['mako_mori_pass'].astype(int)

# Handle "N/A" and convert boxOffice values to numeric
df['boxOffice'] = df['boxOffice'].replace('N/A', np.nan)  # Replace "N/A" with NaN
df['boxOffice'] = df['boxOffice'].str.replace('[\$,]', '', regex=True)  # Remove $ and commas
df['boxOffice'] = pd.to_numeric(df['boxOffice'], errors='coerce')  # Convert to float

# Ensure 'rating' is numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Drop rows with missing values in 'rating' or 'boxOffice' columns
df = df.dropna(subset=['rating', 'boxOffice'])

# Prepare data for rating regression
X_rating = df[['Bechdel', 'Mako_Mori']]
y_rating = df['rating']

# Add a constant for the intercept
X_rating = sm.add_constant(X_rating)

# Ensure that all data in X_rating and y_rating are numeric
X_rating = X_rating.apply(pd.to_numeric)
y_rating = y_rating.apply(pd.to_numeric)

# Fit the OLS regression model for rating
rating_model = sm.OLS(y_rating, X_rating).fit()

print("Rating Regression Results:")
print(rating_model.summary())

# Prepare data for box office regression (financial success)
X_boxOffice = df[['Bechdel', 'Mako_Mori']]
y_boxOffice = df['boxOffice']

# Add a constant for the intercept
X_boxOffice = sm.add_constant(X_boxOffice)

# Ensure all data in X_boxOffice and y_boxOffice are numeric
X_boxOffice = X_boxOffice.apply(pd.to_numeric)
y_boxOffice = y_boxOffice.apply(pd.to_numeric)

# Fit the OLS regression model for box office
box_office_model = sm.OLS(y_boxOffice, X_boxOffice).fit()

print("\nBox Office Regression Results:")
print(box_office_model.summary())

# Calculate Spearman correlation between Bechdel test and rating
bechdel_rating_corr, p_value_bechdel_rating = spearmanr(df['Bechdel'], df['rating'])
print(f"Spearman Correlation between Bechdel Test and Rating: {bechdel_rating_corr}, p-value: {p_value_bechdel_rating}")

# Calculate Spearman correlation between Mako Mori test and rating
mako_mori_rating_corr, p_value_mako_mori_rating = spearmanr(df['Mako_Mori'], df['rating'])
print(f"Spearman Correlation between Mako Mori Test and Rating: {mako_mori_rating_corr}, p-value: {p_value_mako_mori_rating}")

# Calculate Spearman correlation between Bechdel test and box office
bechdel_box_office_corr, p_value_bechdel_box_office = spearmanr(df['Bechdel'], df['boxOffice'])
print(f"Spearman Correlation between Bechdel Test and Box Office: {bechdel_box_office_corr}, p-value: {p_value_bechdel_box_office}")

# Calculate Spearman correlation between Mako Mori test and box office
mako_mori_box_office_corr, p_value_mako_mori_box_office = spearmanr(df['Mako_Mori'], df['boxOffice'])
print(f"Spearman Correlation between Mako Mori Test and Box Office: {mako_mori_box_office_corr}, p-value: {p_value_mako_mori_box_office}")
