"""
Task Description:
                Apply z-score using sklearn library and do mean-centering of Sales dataset(https://www.kaggle.com/datasets/nishathakkar/100-sales) 

Output:
Path to dataset files: /root/.cache/kagglehub/datasets/nishathakkar/100-sales/versions/1
                              Region                Country        Item_Type  \
0              Australia and Oceania                 Tuvalu        Baby Food   
1  Central America and the Caribbean                Grenada           Cereal   
2                             Europe                 Russia  Office Supplies   
3                 Sub_Saharan Africa  Sao Tome and Principe           Fruits   
4                 Sub_Saharan Africa                 Rwanda  Office Supplies   

  Sales_Channel Order_Priority   Ship_Date  Unit_Cost  Total_Revenue  \
0       Offline              H  27/06/2010     159.42     2533654.00   
1        Online              C  15/09/2012     117.11      576782.80   
2       Offline              L  05/08/2014     524.96     1158502.59   
3        Online              C  07/05/2014       6.92       75591.66   
4       Offline              L  02/06/2013     524.96     3296425.02   

   Total_Profit  Unnamed: 9  Unnamed: 10  
0     951410.50         NaN          NaN  
1     248406.36         NaN          NaN  
2     224598.75         NaN          NaN  
3      19525.82         NaN          NaN  
4     639077.50         NaN          NaN

Z-Score Normalized Data:
                              Region                Country        Item_Type  \
0              Australia and Oceania                 Tuvalu        Baby Food   
1  Central America and the Caribbean                Grenada           Cereal   
2                             Europe                 Russia  Office Supplies   
3                 Sub_Saharan Africa  Sao Tome and Principe           Fruits   
4                 Sub_Saharan Africa                 Rwanda  Office Supplies   

  Sales_Channel Order_Priority   Ship_Date  Unit_Cost  Total_Revenue  \
0       Offline              H  27/06/2010  -0.168895       0.798622   
1        Online              C  15/09/2012  -0.394831      -0.548427   
2       Offline              L  05/08/2014   1.783101      -0.147989   
3        Online              C  07/05/2014  -0.983250      -0.893431   
4       Offline              L  02/06/2013   1.783101       1.323690   

   Total_Profit  Unnamed: 9  Unnamed: 10  
0      1.168192         NaN          NaN  
1     -0.442948         NaN          NaN  
2     -0.497510         NaN          NaN  
3     -0.967494         NaN          NaN  
4      0.452390         NaN          NaN 


Mean-Centered Sales Data:
                              Region                Country        Item_Type  \
0              Australia and Oceania                 Tuvalu        Baby Food   
1  Central America and the Caribbean                Grenada           Cereal   
2                             Europe                 Russia  Office Supplies   
3                 Sub_Saharan Africa  Sao Tome and Principe           Fruits   
4                 Sub_Saharan Africa                 Rwanda  Office Supplies   

  Sales_Channel Order_Priority   Ship_Date  Unit_Cost  Total_Revenue  \
0       Offline              H  27/06/2010  -0.168895       0.798622   
1        Online              C  15/09/2012  -0.394831      -0.548427   
2       Offline              L  05/08/2014   1.783101      -0.147989   
3        Online              C  07/05/2014  -0.983250      -0.893431   
4       Offline              L  02/06/2013   1.783101       1.323690   

   Total_Profit  Unnamed: 9  Unnamed: 10  
0      1.168192         NaN          NaN  
1     -0.442948         NaN          NaN  
2     -0.497510         NaN          NaN  
3     -0.967494         NaN          NaN  
4      0.452390         NaN          NaN 

"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("nishathakkar/100-sales")
print("Path to dataset files:", path)

# Load the dataset
file_path = f"{path}/100_Sales.csv"  # Assuming the CSV is named Sales.csv
sales_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(sales_data.head())

# Identify numerical columns (replace with actual column names if known)
numeric_columns = sales_data.select_dtypes(include=['float64', 'int64']).columns

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply Z-score normalization only on numeric columns
sales_data[numeric_columns] = scaler.fit_transform(sales_data[numeric_columns])

# Display the normalized data
print("\nZ-Score Normalized Data:")
print(sales_data.head())


print("\nMean-Centered Sales Data:")
print(sales_data.head())
