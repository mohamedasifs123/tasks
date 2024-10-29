"""
Task Description:
               Implement a custom DataFrame transformation function using `apply` and `groupby`.

Output:

Original DataFrame:
  region  sales month
0  North    200   Jan
1  North    250   Feb
2  South    150   Jan
3  South    100   Feb
4   East    300   Jan
5   East    400   Feb
6   East    350   Mar

Transformed DataFrame:
         region  sales month  cumulative_sales  standardized_sales
region                                                            
East   4   East    300   Jan               300           -1.000000
       5   East    400   Feb               700            1.000000
       6   East    350   Mar              1050            0.000000
North  0  North    200   Jan               200           -0.707107
       1  North    250   Feb               450            0.707107
South  2  South    150   Jan               150            0.707107
       3  South    100   Feb               250           -0.707107
              
"""

import pandas as pd

# Sample Data
data = {
    'region': ['North', 'North', 'South', 'South', 'East', 'East', 'East'],
    'sales': [200, 250, 150, 100, 300, 400, 350],
    'month': ['Jan', 'Feb', 'Jan', 'Feb', 'Jan', 'Feb', 'Mar']
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Custom transformation function to apply to each group
def custom_transformation(group):
    # Calculate cumulative sales
    group['cumulative_sales'] = group['sales'].cumsum()
    
    # Standardize the sales data (mean normalization)
    group['standardized_sales'] = (group['sales'] - group['sales'].mean()) / group['sales'].std()
    
    return group

# Apply the transformation to each group
df_transformed = df.groupby('region').apply(custom_transformation)

print("\nTransformed DataFrame:")
print(df_transformed)

