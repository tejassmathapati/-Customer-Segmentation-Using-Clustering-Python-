import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Bar plot: Total Revenue per Cluster
plt.figure(figsize=(8,5))
sns.barplot(x='Cluster', y='Total_Revenue', data=cluster_profile, palette='viridis')
plt.title('Total Revenue per Cluster')
plt.ylabel('Total Revenue ($)')
plt.xlabel('Cluster')
plt.tight_layout()
plt.show()

# Bar plot: Average Revenue per Customer per Cluster
plt.figure(figsize=(8,5))
sns.barplot(x='Cluster', y='Avg_Revenue_per_Customer', data=cluster_profile, palette='magma')
plt.title('Average Revenue per Customer per Cluster')
plt.ylabel('Avg Revenue per Customer ($)')
plt.xlabel('Cluster')
plt.tight_layout()
plt.show()

# Bar plot: Average Quantity per Transaction per Cluster
plt.figure(figsize=(8,5))
sns.barplot(x='Cluster', y='Avg_Quantity', data=cluster_profile, palette='coolwarm')
plt.title('Average Quantity per Transaction per Cluster')
plt.ylabel('Average Quantity')
plt.xlabel('Cluster')
plt.tight_layout()
plt.show()

# Display cluster profile table
print("\nCluster Profile Table:")
display(cluster_profile.style.format({
    'Total_Revenue': '${:,.2f}',
    'Avg_Revenue_per_Customer': '${:,.2f}',
    'Avg_UnitPrice': '${:,.2f}',
    'Median_UnitPrice': '${:,.2f}'
}))
