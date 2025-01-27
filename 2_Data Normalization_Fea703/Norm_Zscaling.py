# Data normalization
import os
os.chdir("/Users/Zoey/Library/CloudStorage/OneDrive-UniversityofFlorida/PhD_Coding/2_DataProcessing")
os.getcwd()

# Import packages
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
import warnings

# Ignore warnings
warnings.simplefilter("ignore")

# using z-score method (standardization) transforms the info into distribution with a mean of 0 and a deviation of 1. 
# copy the data
swo = pd.read_csv ('SWNS_FS_(no_compound_name)_for_scale.csv', sep=',',encoding='utf-8')
df_z_scaled = swo.copy()
# apply normalizaition techniques
for column in df_z_scaled.columns:
  df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()
# view normalized data
df_z_scaled.head()
df_z_scaled.to_csv("SWNS_FS_zscaled.csv")
                       

