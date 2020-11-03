import pandas as pd
from pandas_profiling import ProfileReport
import time

start_time = time.time()
df = pd.read_csv('train.csv', index_col='Unnamed: 0')
df.index.name ='Ind'
# ---------------------------------------------------------- Cat Encoding
cat_feat = ['type', 'gearbox', 'model', 'fuel', 'brand']
df[cat_feat] = df[cat_feat].astype('category')
for i in cat_feat:
    df[i] = df[i].cat.codes

# ------------------------------------------- Report before selection
profile = ProfileReport(df, title='Profiling Report', explorative=True)
profile.to_file("report(tr).html")
print('running time:', round((time.time() - start_time), 2))
