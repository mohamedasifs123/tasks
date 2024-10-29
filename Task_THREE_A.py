"""
Task Description:
               Find 10000 row dataset and apply all type of normalization in dataset 

Output:

Sample DataFrame:
    feature_1   feature_2  feature_3  feature_4  feature_5
0  104.967142  321.255173  12.435317   0.156382        468
1   98.617357  103.019194   7.184545   1.626021        436
2  106.476885   71.807545  33.181812   0.367508        222
3  115.230299   94.275978   3.644202   1.476648        245
4   97.658466  582.869361  15.366051   1.042893        246


Min-Max Normalized DataFrame (first 5 rows):
   feature_1  feature_2  feature_3  feature_4  feature_5
0   0.563042   0.321247   0.011607   0.014908   0.937876
1   0.482139   0.102984   0.006533   0.155026   0.873747
2   0.582278   0.071768   0.031657   0.035037   0.444890
3   0.693806   0.094239   0.003111   0.140785   0.490982
4   0.469922   0.582893   0.014439   0.099430   0.492986

Z-Score Normalized DataFrame (first 5 rows):
   feature_1  feature_2  feature_3  feature_4  feature_5
0   0.497154  -0.642090  -0.487764  -0.836373   1.510783
1  -0.135665  -1.395950  -0.614622   0.620596   1.287684
2   0.647615  -1.503765   0.013467  -0.627066  -0.204287
3   1.519979  -1.426152  -0.700156   0.472510  -0.043935
4  -0.231228   0.261613  -0.416958   0.042495  -0.036963

Max-Abs Normalized DataFrame (first 5 rows):
   feature_1  feature_2  feature_3  feature_4  feature_5
0   0.753737   0.321279   0.012013   0.014910   0.937876
1   0.708141   0.103027   0.006940   0.155028   0.873747
2   0.764578   0.071813   0.032054   0.035039   0.444890
3   0.827433   0.094283   0.003520   0.140786   0.490982
4   0.701255   0.582913   0.014844   0.099431   0.492986

Robust Scaled DataFrame (first 5 rows):
   feature_1  feature_2  feature_3  feature_4  feature_5
0   0.371601  -0.380287  -0.259360  -0.473329   0.866397
1  -0.100969  -0.817057  -0.438464   0.851643   0.736842
2   0.483960  -0.879523   0.448304  -0.282985  -0.129555
3   1.135415  -0.834555  -0.559225   0.716974  -0.036437
4  -0.172333   0.143298  -0.159392   0.325916  -0.032389

"""
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset with 10,000 rows and 5 features
data = {
    'feature_1': np.random.normal(100, 10, 10000),  # Normal distribution
    'feature_2': np.random.uniform(0, 1000, 10000),  # Uniform distribution
    'feature_3': np.random.lognormal(3, 1, 10000),  # Log-normal distribution
    'feature_4': np.random.exponential(1, 10000),    # Exponential distribution
    'feature_5': np.random.randint(0, 500, 10000)    # Random integers
}

# Create DataFrame
df = pd.DataFrame(data)

print("Sample DataFrame:")
print(df.head())

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# Initialize scalers
min_max_scaler = MinMaxScaler()
z_score_scaler = StandardScaler()
max_abs_scaler = MaxAbsScaler()
robust_scaler = RobustScaler()

# Apply Min-Max Normalization
df_min_max_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
print("\nMin-Max Normalized DataFrame (first 5 rows):")
print(df_min_max_scaled.head())

# Apply Z-Score Normalization
df_z_score_scaled = pd.DataFrame(z_score_scaler.fit_transform(df), columns=df.columns)
print("\nZ-Score Normalized DataFrame (first 5 rows):")
print(df_z_score_scaled.head())

# Apply Max Abs Normalization
df_max_abs_scaled = pd.DataFrame(max_abs_scaler.fit_transform(df), columns=df.columns)
print("\nMax-Abs Normalized DataFrame (first 5 rows):")
print(df_max_abs_scaled.head())

# Apply Robust Scaling
df_robust_scaled = pd.DataFrame(robust_scaler.fit_transform(df), columns=df.columns)
print("\nRobust Scaled DataFrame (first 5 rows):")
print(df_robust_scaled.head())
