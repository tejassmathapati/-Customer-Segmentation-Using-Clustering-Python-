
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


file_path = 'Online Retail.xlsx'
df = pd.read_excel(file_path)


print("First 5 rows of the dataset:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())

df_clean = df.dropna(subset=['CustomerID'])

df_clean = df_clean.drop_duplicates()

df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

print("\nData shape after cleaning:", df_clean.shape)

print("\nSummary statistics:")
print(df_clean.describe())
num_customers = df_clean['CustomerID'].nunique()
print(f"\nNumber of unique customers: {num_customers}")
num_products = df_clean['StockCode'].nunique()
print(f"Number of unique products: {num_products}")

top_products_qty = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 products by quantity sold:")
print(top_products_qty)

df_clean['Revenue'] = df_clean['Quantity'] * df_clean['UnitPrice']
top_products_revenue = df_clean.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 products by revenue:")
print(top_products_revenue)

plt.figure(figsize=(10,5))
sns.histplot(df_clean['Quantity'], bins=50, kde=False)
plt.title('Distribution of Quantity')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.xlim(0, 100)  # limit x-axis for better visualization
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df_clean['UnitPrice'], bins=50, kde=False)
plt.title('Distribution of Unit Price')
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
plt.xlim(0, 100)  # limit x-axis for better visualization
plt.show()

df_clean.set_index('InvoiceDate', inplace=True)
monthly_revenue = df_clean['Revenue'].resample('M').sum()

plt.figure(figsize=(12,6))
monthly_revenue.plot()
plt.title('Monthly Revenue Over Time')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.show()

monthly_invoices = df_clean['InvoiceNo'].resample('M').nunique()

plt.figure(figsize=(12,6))
monthly_invoices.plot()
plt.title('Number of Invoices Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Invoices')
plt.show()



cols_to_scale = ['Quantity', 'UnitPrice', 'Revenue']

data_to_scale = df_clean[cols_to_scale]

min_max_scaler = MinMaxScaler()
data_normalized = min_max_scaler.fit_transform(data_to_scale)

df_normalized = pd.DataFrame(data_normalized, columns=[col + '_normalized' for col in cols_to_scale])

standard_scaler = StandardScaler()
data_scaled = standard_scaler.fit_transform(data_to_scale)

df_scaled = pd.DataFrame(data_scaled, columns=[col + '_scaled' for col in cols_to_scale])

df_final = pd.concat([df_clean.reset_index(drop=True), df_normalized, df_scaled], axis=1)

print(df_final.head())

