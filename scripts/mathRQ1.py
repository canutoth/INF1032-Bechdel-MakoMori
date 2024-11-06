import pandas as pd
import scipy.stats as stats
import numpy as np

data = pd.read_csv('data.csv')

both = data['both'].sum()
neither = data['neither'].sum()
bechdel_only = data['bechdel'].sum()
mako_mori_only = data['makomori'].sum()

contingency_table = np.array([[both, bechdel_only], [mako_mori_only, neither]])

print("Contingency Table:")
print(contingency_table)

chi2, p, _, _ = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:\nChi2 = {chi2}, p-value = {p}")

if p < 0.05:
    phi_coefficient = np.sqrt(chi2 / np.sum(contingency_table))
    print(f"\nPhi Coefficient: {phi_coefficient}")
else:
    print("\nNo significant association between Bechdel and Mako Mori based on Chi-Square test (p > 0.05).")


odds_ratio = (both * neither) / (bechdel_only * mako_mori_only)
log_or = np.log(odds_ratio)
se_log_or = np.sqrt(1/both + 1/neither + 1/bechdel_only + 1/mako_mori_only)

ci_lower = np.exp(log_or - 1.96 * se_log_or)
ci_upper = np.exp(log_or + 1.96 * se_log_or)

print(f"\nOdds Ratio: {odds_ratio}")
print(f"95% Confidence Interval for Odds Ratio: ({ci_lower}, {ci_upper})")