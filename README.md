# Predicting-House-Prices-using-Machine-Learning-###           Predicting-House-Prices-Using-Machine-Learning
## 1.Introduction
    Briefly explain the importance of predicting house prices. State the purpose and scope of the document.
  # Importing Libraries and Dataset:
           Matplotlib 
           Seaborn
           Pandas
## 2.Data Preprocessing:
```bash
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))
```
## 3.Exploratory Data Analysis:
```bash
plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(),
cmap = 'BrBG',
fmt = '.2f',
linewidths = 2,
annot = True)
```
## 4.Data Cleaning:
```bash
dataset.drop(['Id'],axis=1,inplace=True)
```
## 5.OneHotEncoder â€“ For Label categorical features:
```bash
from sklearn.preprocessing import OneHotEncoder
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', 
len(object_cols))
```
## 6.Splitting Dataset into Training and Testing:
 ```bash
 X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, 
 random_state=101)
 Y_train.head()
```
## 7.Standardizing the data:
```bash
Sc = StandardScaler()
X_train_scal = sc.fit_transform(X_train)
X_test_scal = sc.fit_transform(X_test)
```
## 8.Predicting Prices:
```bash
Prediction1 = model_lr.predict(X_test_scal)
```
## 9.Evaluation of Predicted Data:
```bash
Plt.figure(figsize=(12,6))
Plt.plot(np.arange(len(Y_test)), Y_test, label=â€™Actual Trendâ€™)
Plt.plot(np.arange(len(Y_test)), Prediction1, label=â€™Predicted Trendâ€™)
Plt.xlabel(â€˜Dataâ€™)
Plt.ylabel(â€˜Trendâ€™)
Plt.legend()
Plt.title(â€˜Actual vs Predictedâ€™)
```
