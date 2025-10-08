# Cluster Profiling

# Group by cluster and calculate summary statistics
cluster_profile = df_final.groupby('Cluster').agg({
    'Quantity': ['mean', 'median', 'sum'],
    'UnitPrice': ['mean', 'median'],
    'Revenue': ['mean', 'median', 'sum'],
    'CustomerID': 'nunique',
    'InvoiceNo': 'nunique'
}).reset_index()

# Rename columns for clarity
cluster_profile.columns = ['Cluster',
                           'Avg_Quantity', 'Median_Quantity', 'Total_Quantity',
                           'Avg_UnitPrice', 'Median_UnitPrice',
                           'Avg_Revenue', 'Median_Revenue', 'Total_Revenue',
                           'Unique_Customers', 'Unique_Invoices']

print("Cluster Profile Summary:")
print(cluster_profile)

# Additional insights: average revenue per customer
cluster_profile['Avg_Revenue_per_Customer'] = cluster_profile['Total_Revenue'] / cluster_profile['Unique_Customers']

print("\nCluster Profile with Avg Revenue per Customer:")
print(cluster_profile)

# Draft Cluster Summary Report and Marketing Recommendations

for idx, row in cluster_profile.iterrows():
    print(f"\n--- Cluster {int(row['Cluster'])} Summary ---")
    print(f"Number of unique customers: {row['Unique_Customers']}")
    print(f"Total revenue: ${row['Total_Revenue']:.2f}")
    print(f"Average revenue per customer: ${row['Avg_Revenue_per_Customer']:.2f}")
    print(f"Average quantity per transaction: {row['Avg_Quantity']:.2f}")
    print(f"Average unit price: ${row['Avg_UnitPrice']:.2f}")
    
    # Marketing recommendations based on spending behavior
    if row['Avg_Revenue_per_Customer'] > cluster_profile['Avg_Revenue_per_Customer'].mean():
        print("Marketing Strategy: High-value customers. Recommend loyalty programs, exclusive offers, and premium product promotions.")
    elif row['Avg_Revenue_per_Customer'] > cluster_profile['Avg_Revenue_per_Customer'].median():
        print("Marketing Strategy: Mid-value customers. Recommend targeted discounts, bundle offers, and personalized recommendations.")
    else:
        print("Marketing Strategy: Low-value customers. Recommend awareness campaigns, introductory discounts, and engagement through newsletters.")
