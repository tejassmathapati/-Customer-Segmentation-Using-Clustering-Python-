# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
file_path = 'Online Retail.xlsx'
df = pd.read_excel(file_path)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Data Cleaning

# 1. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 2. Remove rows with missing CustomerID (important for customer analysis)
df_clean = df.dropna(subset=['CustomerID'])

# 3. Remove duplicates
df_clean = df_clean.drop_duplicates()

# 4. Convert InvoiceDate to datetime if not already
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# 5. Remove rows with negative or zero Quantity or UnitPrice (assuming these are invalid for sales)
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

# Summary after cleaning
print("\nData shape after cleaning:", df_clean.shape)

# Exploratory Data Analysis (EDA)

# 1. Summary statistics
print("\nSummary statistics:")
print(df_clean.describe())

# 2. Number of unique customers
num_customers = df_clean['CustomerID'].nunique()
print(f"\nNumber of unique customers: {num_customers}")

# 3. Number of unique products
num_products = df_clean['StockCode'].nunique()
print(f"Number of unique products: {num_products}")

# 4. Top 10 products by quantity sold
top_products_qty = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 products by quantity sold:")
print(top_products_qty)

# 5. Top 10 products by revenue
df_clean['Revenue'] = df_clean['Quantity'] * df_clean['UnitPrice']
top_products_revenue = df_clean.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 products by revenue:")
print(top_products_revenue)

# 6. Plot distribution of Quantity
plt.figure(figsize=(10,5))
sns.histplot(df_clean['Quantity'], bins=50, kde=False)
plt.title('Distribution of Quantity')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.xlim(0, 100)  # limit x-axis for better visualization
plt.show()

# 7. Plot distribution of UnitPrice
plt.figure(figsize=(10,5))
sns.histplot(df_clean['UnitPrice'], bins=50, kde=False)
plt.title('Distribution of Unit Price')
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
plt.xlim(0, 100)  # limit x-axis for better visualization
plt.show()

# 8. Time series analysis: total revenue per month
df_clean.set_index('InvoiceDate', inplace=True)
monthly_revenue = df_clean['Revenue'].resample('M').sum()

plt.figure(figsize=(12,6))
monthly_revenue.plot()
plt.title('Monthly Revenue Over Time')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.show()

# 9. Number of invoices over time
monthly_invoices = df_clean['InvoiceNo'].resample('M').nunique()

plt.figure(figsize=(12,6))
monthly_invoices.plot()
plt.title('Number of Invoices Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Invoices')
plt.show()



# Assuming df_clean is your cleaned DataFrame and has 'Quantity', 'UnitPrice', 'Revenue'

# Select columns to scale
cols_to_scale = ['Quantity', 'UnitPrice', 'Revenue']

# Extract the data to scale
data_to_scale = df_clean[cols_to_scale]

# 1. Min-Max Normalization (scales data to [0,1])
min_max_scaler = MinMaxScaler()
data_normalized = min_max_scaler.fit_transform(data_to_scale)

# Convert back to DataFrame
df_normalized = pd.DataFrame(data_normalized, columns=[col + '_normalized' for col in cols_to_scale])

# 2. Standardization (zero mean, unit variance)
standard_scaler = StandardScaler()
data_scaled = standard_scaler.fit_transform(data_to_scale)

# Convert back to DataFrame
df_scaled = pd.DataFrame(data_scaled, columns=[col + '_scaled' for col in cols_to_scale])

# Concatenate normalized and scaled columns to original DataFrame
df_final = pd.concat([df_clean.reset_index(drop=True), df_normalized, df_scaled], axis=1)

# Display first few rows of the final DataFrame
print(df_final.head())
