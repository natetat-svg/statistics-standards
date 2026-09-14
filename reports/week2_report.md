import pandas as pd

ht_df = pd.read_csv('reports/hypothesis_tests_summary.csv')
dist_df = pd.read_csv('reports/distribution_fitting_summary.csv')
ci_df = pd.read_csv('reports/ci_comparison.csv')

print("=== HYPOTHESIS TESTS ===")
print(ht_df.to_markdown(index=False))

print("\n=== DISTRIBUTION FITTING ===")
print(dist_df.to_markdown(index=False))

print("\n=== CONFIDENCE INTERVALS ===")
print(ci_df.to_markdown(index=False))
