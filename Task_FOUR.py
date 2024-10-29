"""
Task Description:
               Work with multi-index DataFrames and perform operations on different levels.

Output:

Multi-Index DataFrame:
                    sales  profit
region city   year               
North  City A 2020    200      50
       City B 2020    150      30
South  City A 2020    300      70
       City B 2020    100      20
East   City A 2021    400      90
       City B 2021    350      80

Data for North region:
             sales  profit
city   year               
City A 2020    200      50
City B 2020    150      30

Mean sales and profit for each region:
        sales  profit
region               
East    375.0    85.0
North   175.0    40.0
South   200.0    45.0

"""

import pandas as pd

# Define data
data = {
    'region': ['North', 'North', 'South', 'South', 'East', 'East'],
    'city': ['City A', 'City B', 'City A', 'City B', 'City A', 'City B'],
    'year': [2020, 2020, 2020, 2020, 2021, 2021],
    'sales': [200, 150, 300, 100, 400, 350],
    'profit': [50, 30, 70, 20, 90, 80]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set multi-index: region, city, year
df_multi = df.set_index(['region', 'city', 'year'])

print("Multi-Index DataFrame:")
print(df_multi)

# Access all data for the "North" region
north_region = df_multi.loc['North']
print("Data for North region:")
print(north_region)

# Reset index to turn multi-index into columns again
df_reset = df_multi.reset_index()
print("Reset index DataFrame:")
print(df_reset)

# Group by region and calculate the mean for sales and profit
region_mean = df_multi.groupby(level='region').mean()
print("Mean sales and profit for each region:")
print(region_mean)
