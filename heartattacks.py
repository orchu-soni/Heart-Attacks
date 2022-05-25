import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from google.colab import files
import io
uploaded=files.upload()
data1 = pd.read_csv(io.BytesIO(uploaded["heart.csv"]))
data1 = pd.DataFrame(data1)


data1.info()
data1.shape


duplicate_rows = data1[data1.duplicated()]
print("Number of duplicate rows :: ", duplicate_rows.shape)


data1 = data1.drop_duplicates()
duplicate_rows = data1[data1.duplicated()]
print("Number of duplicate rows :: ", duplicate_rows.shape)

print("Null values :: ")
print(data1.isnull() .sum())
data1.shape

sns.boxplot(x=data1['age'])
#No Outliers observed in 'age'
sns.boxplot(x=data1['sex'])
#No outliers observed in sex data
sns.boxplot(x=data1['cp'])
#No outliers in 'cp'
sns.boxplot(x=data1['trtbps'])
#Some outliers are observed in 'trtbps'. They will be removed later
sns.boxplot(x=data1['chol'])
#Some outliers are observed in 'chol'. They will be removed later
sns.boxplot(x=data1['fbs'])
sns.boxplot(x=data1['restecg'])
sns.boxplot(x=data1['thalachh'])
#Outliers present in thalachh
sns.boxplot(x=data1['exng'])
sns.boxplot(x=data1['oldpeak'])
#Outliers are present in 'OldPeak'
sns.boxplot(x=data1['slp'])
sns.boxplot(x=data1['caa'])
#Outliers are present in 'caa'
sns.boxplot(x=data1['thall'])


Q1 = data1.quantile(0.25)
Q3 = data1.quantile(0.75)
IQR = Q3-Q1
print('*********** InterQuartile Range ***********')
print(IQR)
# Remove the outliers using IQR
data2 = data1[~((data1<(Q1-1.5*IQR))|(data1>(Q3+1.5*IQR))).any(axis=1)]
data2.shape


z = np.abs(stats.zscore(data1))
data3 = data1[(z<3).all(axis=1)]
data3.shape

pearsonCorr = data3.corr(method='pearson')
spearmanCorr = data3.corr(method='spearman')
fig = plt.subplots(figsize=(14,8))
sns.heatmap(pearsonCorr, vmin=-1,vmax=1, cmap = "Greens", annot=True, linewidth=0.1)
plt.title("Pearson Correlation")
