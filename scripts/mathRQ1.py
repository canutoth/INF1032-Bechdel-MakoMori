import pandas as pd
import scipy.stats as stats
import numpy as np
import json
import statsmodels.api as sm

with open('summedUp_data.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

df['bechdel_pass'] = df['bechdel'] >= 3
df['mako_mori_pass'] = df['mako-mori'] > 0

df['both'] = df['bechdel_pass'] & df['mako_mori_pass']
df['neither'] = ~df['bechdel_pass'] & ~df['mako_mori_pass']
df['bechdel_only'] = df['bechdel_pass'] & ~df['mako_mori_pass']
df['mako_mori_only'] = ~df['bechdel_pass'] & df['mako_mori_pass']

both_count = df['both'].sum()
neither_count = df['neither'].sum()
bechdel_only_count = df['bechdel_only'].sum()
mako_mori_only_count = df['mako_mori_only'].sum()

contingency_table = np.array([[both_count, bechdel_only_count], 
                              [mako_mori_only_count, neither_count]])

print("Contingency Table:")
print(contingency_table)

chi2, p, _, _ = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:\nChi2 = {chi2}, p-value = {p}")

if p < 0.05:
    phi_coefficient = np.sqrt(chi2 / np.sum(contingency_table))
    print(f"\nPhi Coefficient: {phi_coefficient}")
else:
    print("\nNo significant association between Bechdel and Mako Mori based on Chi-Square test (p > 0.05).")

odds_ratio = (both_count * neither_count) / (bechdel_only_count * mako_mori_only_count)
log_or = np.log(odds_ratio)
se_log_or = np.sqrt(1/both_count + 1/neither_count + 1/bechdel_only_count + 1/mako_mori_only_count)

ci_lower = np.exp(log_or - 1.96 * se_log_or)
ci_upper = np.exp(log_or + 1.96 * se_log_or)

print(f"\nOdds Ratio: {odds_ratio}")
print(f"95% Confidence Interval for Odds Ratio: ({ci_lower}, {ci_upper})")