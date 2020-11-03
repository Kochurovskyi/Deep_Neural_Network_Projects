import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
import time

def mean_absolute_percentage_error(y_true, y_pred):             # error function
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


start_time = time.time()
df = pd.read_csv('train.csv').drop('Unnamed: 0', 1)
#print(df.info())
# ---------------------------------------------------------- fillna

df.type = df['type'].fillna('NO')
df.gearbox = df['gearbox'].fillna('NO')
df.fuel = df['fuel'].fillna('NO')
df.model = df['model'].fillna('NO')


df = df.dropna(axis=0) #----------------------------------- drop na
# ---------------------------------------------------------- Cat Encoding
cat_feat = ['type', 'gearbox', 'model', 'fuel', 'brand']
df[cat_feat] = df[cat_feat].astype('category')
for i in cat_feat:
    df[i] = df[i].cat.codes
# --------------------------------------------------------- Cormatrix
'''
f, ax = plt.subplots(figsize=(15, 15))
ax = sp.heatmap(df.corr(),
                annot=True, vmin=-1, vmax=1,
                center=0, cmap= 'coolwarm',
                linewidths=1, linecolor='black')
plt.show()
'''


#df = df.iloc[:150, :]
print(df.info())
X = df.drop('price', 1)
y = df.price
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)
model = XGBClassifier()
model.fit(X_tr, y_tr)
pred = model.predict(X_ts)
#print('XGB feature_importances_: ', model.feature_importances_)  # -- XGB Feature importance
plt.bar(X.columns, model.feature_importances_) # -------- XGB Feature importance Plot
print('Start MAPE:', np.round(mean_absolute_percentage_error(pred, y_ts),3))
print('running time:', round((time.time() - start_time), 2))
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


# ------------------------------------------------------------------------ ADD run
feat_lst = list(X.columns)
res_lst = []
best_scr = 1000
for i in feat_lst:              # Add run
    res_lst.append(i)
    print('Check:', i)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X[res_lst], y, test_size=0.3, random_state=0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_ts)
    cur_scr = mean_absolute_percentage_error(pred, y_ts)
    print('Num.Feat:{} --> MAPE:{}'.format(len(res_lst), np.round(cur_scr, 2)))
    print('running time:', round((time.time() - start_time), 2))
    if cur_scr < best_scr:
        best_scr = cur_scr
        print('Status: Added')
    else:
        print(res_lst[-1], 'Status: passed')
        del res_lst[-1]
    print('-----------------')
print('Add results:', res_lst)
print('best_scr:', round(best_scr, 2))
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

# ------------------------------------------------------------------------ DEL Run
feat_lst = res_lst
for i in feat_lst[:-1]:               # Dell run
    res_lst.remove(i)
    print('Check:', i)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X[res_lst], y, test_size=0.3, random_state=0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_ts)
    cur_scr = mean_absolute_percentage_error(pred, y_ts)
    print('Num.Feat:{} --> MAPE:{}'.format(len(res_lst), np.round(cur_scr, 2)))
    print('running time:', round((time.time() - start_time), 2))
    if cur_scr < best_scr:
        best_scr = cur_scr
        print('Status: Deleted')
    else:
        print('Status: Left')
        res_lst.append(i)
    print('-----------------')
print('Del results:', res_lst)
print('best_scr:', round(best_scr, 2))
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

print('running time:', round((time.time() - start_time),2))

