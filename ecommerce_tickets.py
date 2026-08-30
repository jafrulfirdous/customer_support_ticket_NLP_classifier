import pandas as pd
import numpy as np
from pathlib import Path

# Load the CSV file
csv_path = Path(__file__).parent / "customer_support_tickets.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Dataset shape: {df.shape}")
print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
print("\nColumn names:")
print(df.columns.tolist())

# Analyze missing values
print("\n" + "=" * 80)
print("MISSING VALUES ANALYSIS")
print("=" * 80)
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_data.to_string(index=False))

# Display sample data
print("\n" + "=" * 80)
print("SAMPLE DATA (First 5 rows)")
print("=" * 80)
print(df.head())

# ============================================================================
# HANDLE MISSING VALUES
# ============================================================================
print("\n" + "=" * 80)
print("HANDLING MISSING VALUES")
print("=" * 80)

df_cleaned = df.copy()

# 1. Handle numeric columns - fill with median
print("\n1. Filling numeric columns with median:")
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_cleaned[col].isnull().sum() > 0:
        median_val = df_cleaned[col].median()
        missing_count = df[col].isnull().sum()
        df_cleaned[col] = df_cleaned[col].fillna(median_val)
        print(f"   - {col}: Filled {missing_count} missing values with median ({median_val})")

# 2. Handle categorical columns - fill with mode (most common value)
print("\n2. Filling categorical columns with mode (most common value):")
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_cleaned[col].isnull().sum() > 0:
        mode_val = df_cleaned[col].mode()
        if len(mode_val) > 0:
            mode_val = mode_val[0]
            missing_count = df[col].isnull().sum()
            df_cleaned[col] = df_cleaned[col].fillna(mode_val)
            print(f"   - {col}: Filled {missing_count} missing values with mode ('{mode_val}')")

# 3. Handle specific columns with custom logic
print("\n3. Custom handling for specific columns:")

# Fill 'First Response Time' with a default value if empty
if 'First Response Time' in df_cleaned.columns:
    frt_null = df_cleaned['First Response Time'].isnull().sum()
    df_cleaned['First Response Time'] = df_cleaned['First Response Time'].fillna('Not Yet Responded')
    print(f"   - First Response Time: Filled {frt_null} values with 'Not Yet Responded'")

# Fill 'Time to Resolution' with a default value if empty
if 'Time to Resolution' in df_cleaned.columns:
    ttr_null = df_cleaned['Time to Resolution'].isnull().sum()
    df_cleaned['Time to Resolution'] = df_cleaned['Time to Resolution'].fillna('Pending')
    print(f"   - Time to Resolution: Filled {ttr_null} values with 'Pending'")

# Fill 'Customer Satisfaction Rating' with 0 (indicating not rated)
if 'Customer Satisfaction Rating' in df_cleaned.columns:
    csr_null = df_cleaned['Customer Satisfaction Rating'].isnull().sum()
    df_cleaned['Customer Satisfaction Rating'] = df_cleaned['Customer Satisfaction Rating'].fillna(0)
    print(f"   - Customer Satisfaction Rating: Filled {csr_null} values with 0 (Not Rated)")

# Fill 'Resolution' with a default value
if 'Resolution' in df_cleaned.columns:
    res_null = df_cleaned['Resolution'].isnull().sum()
    df_cleaned['Resolution'] = df_cleaned['Resolution'].fillna('No Resolution Provided')
    print(f"   - Resolution: Filled {res_null} values with 'No Resolution Provided'")

# Verify no missing values remain
print("\n" + "=" * 80)
print("VERIFICATION - MISSING VALUES AFTER HANDLING")
print("=" * 80)
remaining_missing = df_cleaned.isnull().sum()
if remaining_missing.sum() == 0:
    print("[OK] No missing values remaining! Dataset is clean.")
else:
    print("[WARNING] Some missing values still remain:")
    print(remaining_missing[remaining_missing > 0])

# Save the cleaned dataset
output_path = Path(__file__).parent / "customer_support_tickets_cleaned.csv"
df_cleaned.to_csv(output_path, index=False)
print(f"\n[OK] Cleaned dataset saved to: {output_path}")

# Display summary statistics
print("\n" + "=" * 80)
print("CLEANED DATASET SUMMARY")
print("=" * 80)
print(df_cleaned.head(10))


