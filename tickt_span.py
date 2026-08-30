import pandas as pd
import numpy as np

# Read the raw CSV file
df = pd.read_csv('customer_support_tickets.csv')

print("\n" + "="*70)
print("TICKET LIFESPAN ANALYSIS - Resolution Time")
print("="*70)

# Convert date columns to datetime
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], errors='coerce')

# Calculate lifespan (time from purchase to resolution) in hours
df['Lifespan_Hours'] = (df['Time to Resolution'] - df['Date of Purchase']).dt.total_seconds() / 3600

# Remove negative values (invalid data)
df['Lifespan_Hours'] = df['Lifespan_Hours'].apply(lambda x: np.nan if x < 0 else x)

print(f"\nTotal tickets: {len(df)}")
print(f"Tickets with resolution time: {df['Lifespan_Hours'].notna().sum()}")
print(f"Tickets without resolution: {df['Lifespan_Hours'].isna().sum()}")

# Basic statistics
print("\n" + "-"*70)
print("LIFESPAN STATISTICS (Hours)")
print("-"*70)
print(f"Mean: {df['Lifespan_Hours'].mean():.2f} hours ({df['Lifespan_Hours'].mean()/24:.2f} days)")
print(f"Median: {df['Lifespan_Hours'].median():.2f} hours ({df['Lifespan_Hours'].median()/24:.2f} days)")
print(f"Min: {df['Lifespan_Hours'].min():.2f} hours")
print(f"Max: {df['Lifespan_Hours'].max():.2f} hours")
print(f"Std Dev: {df['Lifespan_Hours'].std():.2f} hours")

# Detect outliers using IQR method
Q1 = df['Lifespan_Hours'].quantile(0.25)
Q3 = df['Lifespan_Hours'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (df['Lifespan_Hours'] < lower_bound) | (df['Lifespan_Hours'] > upper_bound)

print("\n" + "-"*70)
print("OUTLIER DETECTION (IQR Method)")
print("-"*70)
print(f"Q1 (25th percentile): {Q1:.2f} hours")
print(f"Q3 (75th percentile): {Q3:.2f} hours")
print(f"IQR: {IQR:.2f} hours")
print(f"Lower bound: {lower_bound:.2f} hours")
print(f"Upper bound: {upper_bound:.2f} hours")
print(f"\nOutliers found: {outliers.sum()} tickets ({(outliers.sum()/len(df)*100):.2f}%)")

# Show outlier tickets
if outliers.sum() > 0:
    print("\n" + "-"*70)
    print("OUTLIER TICKETS (Long resolution times)")
    print("-"*70)
    outlier_tickets = df[outliers][['Ticket ID', 'Ticket Status', 'Ticket Priority', 'Lifespan_Hours']].sort_values('Lifespan_Hours', ascending=False)
    print(outlier_tickets.to_string(index=False))

# Remove outliers and show cleaned statistics
df_cleaned = df[~outliers].copy()
print("\n" + "-"*70)
print("STATISTICS AFTER REMOVING OUTLIERS")
print("-"*70)
print(f"Cleaned tickets: {len(df_cleaned)}")
print(f"Mean: {df_cleaned['Lifespan_Hours'].mean():.2f} hours ({df_cleaned['Lifespan_Hours'].mean()/24:.2f} days)")
print(f"Median: {df_cleaned['Lifespan_Hours'].median():.2f} hours ({df_cleaned['Lifespan_Hours'].median()/24:.2f} days)")
print(f"Min: {df_cleaned['Lifespan_Hours'].min():.2f} hours")
print(f"Max: {df_cleaned['Lifespan_Hours'].max():.2f} hours")

# Analysis by ticket status
print("\n" + "-"*70)
print("AVERAGE LIFESPAN BY TICKET STATUS")
print("-"*70)
status_stats = df.groupby('Ticket Status')['Lifespan_Hours'].agg(['count', 'mean', 'median']).round(2)
status_stats.columns = ['Count', 'Mean (Hours)', 'Median (Hours)']
print(status_stats)

# Analysis by priority
print("\n" + "-"*70)
print("AVERAGE LIFESPAN BY TICKET PRIORITY")
print("-"*70)
priority_stats = df.groupby('Ticket Priority')['Lifespan_Hours'].agg(['count', 'mean', 'median']).round(2)
priority_stats.columns = ['Count', 'Mean (Hours)', 'Median (Hours)']
print(priority_stats)

# Save results
df[['Ticket ID', 'Ticket Status', 'Ticket Priority', 'Date of Purchase', 'Time to Resolution', 'Lifespan_Hours']].to_csv('ticket_lifespan_results.csv', index=False)
print("\n" + "="*70)
print("✓ Results saved to 'ticket_lifespan_results.csv'")
print("="*70)
