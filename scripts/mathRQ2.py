import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('data.csv')

data['Bechdel'] = data.apply(lambda x: 1 if x['both'] > 0 or x['bechdel'] > 0 else 0, axis=1)
data['Mako_Mori'] = data.apply(lambda x: 1 if x['both'] > 0 or x['makomori'] > 0 else 0, axis=1)

X_rating = data[['Bechdel', 'Mako_Mori']]
y_rating = data['rating']

X_rating = sm.add_constant(X_rating)

rating_model = sm.OLS(y_rating, X_rating).fit()

print("Rating Regression Results:")
print(rating_model.summary())

X_profit = data[['Bechdel', 'Mako_Mori']]
y_profit = data['profit']

X_profit = sm.add_constant(X_profit)

profit_model = sm.OLS(y_profit, X_profit).fit()

print("\nProfit Regression Results:")
print(profit_model.summary())
