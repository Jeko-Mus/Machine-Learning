import numpy    as np
from numpy.testing._private.utils import decorate_methods
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl
import time

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config

from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.ensemble      import ExtraTreesRegressor
from sklearn.ensemble      import AdaBoostRegressor
from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from catboost              import CatBoostRegressor

data = pd.read_csv('london_merged.csv')

np.random.seed(0)
data.head()
data.isnull().sum()
data.info()
data.describe()
data.weather_code.unique()

# looks like t1 and t2 highly correlated thus unecesary to have them both in the dataset
data_num = data.drop(['timestamp'], axis=1)
fig, ax = plt.subplots(figsize=(8,5))  
sb.heatmap(data_num.corr(), cmap='YlGnBu', annot=True) #YlGnBu viridis
plt.show()

# makes sense to split timestamp into year and month and hour for better analysis and also because bike share data is per hour. Will also drop t1 in following cell as 'real feel' matters more

data['year'] = data['timestamp'].apply(lambda row: row[:4])
data['month'] = data['timestamp'].apply(lambda row: row.split('-')[2][:2] )
data['hour'] = data['timestamp'].apply(lambda row: row.split(':')[0][-2:] )

data.drop(['timestamp'], axis=1, inplace=True)
data.drop(['t1'], axis=1, inplace=True)
data.head()

data.dtypes

def data_enhancement(data):
    
    gen_data = data
    for weather_code in data.weather_code.unique():
        weather_data =  gen_data[gen_data['weather_code'] == weather_code]
        
        hum_cv = weather_data['hum'].std() /weather_data['hum'].mean()
        wind_speed_cv = weather_data['wind_speed'].std() / weather_data['wind_speed'].mean()
        t2_cv = weather_data['t2'].std() / weather_data['t2'].mean()

        for i in gen_data[gen_data['weather_code'] == weather_code].index:
            if np.random.randint(2) == 1:
                gen_data['hum'].values[i] += hum_cv 
            else:
                gen_data['hum'].values[i] -= hum_cv 
                
            if np.random.randint(2) == 1:
                gen_data['wind_speed'].values[i] += wind_speed_cv 
            else:
                gen_data['wind_speed'].values[i] -= wind_speed_cv
                
            if np.random.randint(2) == 1:
                gen_data['t2'].values[i] += t2_cv
            else:
                gen_data['t2'].values[i] -= t2_cv

    return gen_data


data.head(3)
gen = data_enhancement(data)
gen.head(3)

# create season weekend variable

# percentage error went up when i used this
#data['season_weekend'] = data['season'] * data['is_weekend']

# generate x and y 

y = data['cnt']
x = data.drop(['cnt'], axis=1)

cat_vars = ['year','month','hour', 'weather_code','is_weekend','is_holiday','season']
#cat_vars = ['year','month','hour', 'weather_code','season_weekend','is_holiday','season']
num_vars = ['t2','hum','wind_speed']

x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)

extra_sample = gen.sample(gen.shape[0] // 4)
x_train = pd.concat([x_train, extra_sample.drop(['cnt'], axis=1 ) ])
y_train = pd.concat([y_train, extra_sample['cnt'] ])

transformer = preprocessing.PowerTransformer()
y_train = transformer.fit_transform(y_train.values.reshape(-1,1))
y_val = transformer.transform(y_val.values.reshape(-1,1))

rang = abs(y_train.max()) + abs(y_train.min())

num_4_treeModels = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='constant', fill_value=-9999)),
])

cat_4_treeModels = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='constant', fill_value='missing')),
    ('ordinal', preprocessing.OrdinalEncoder()) # handle_unknown='ignore' ONLY IN VERSION 0.24
])

tree_prepro = compose.ColumnTransformer(transformers=[
    ('num', num_4_treeModels, num_vars),
    ('cat', cat_4_treeModels, cat_vars),
], remainder='drop') # Drop other vars not specified in num_vars or cat_vars

tree_classifiers = {
  "Decision Tree": DecisionTreeRegressor(),
  "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
  "Random Forest": RandomForestRegressor(n_estimators=100),
  "AdaBoost":      AdaBoostRegressor(n_estimators=100),
  "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
  "XGBoost":       XGBRegressor(n_estimators=100),
  "LightGBM":      LGBMRegressor(n_estimators=100),
  "CatBoost":      CatBoostRegressor(n_estimators=100),
}

tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}

results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})

for model_name, model in tree_classifiers.items():
    
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_val)
    
    results = results.append({"Model":    model_name,
                              "MSE": metrics.mean_squared_error(y_val, pred),
                              "MAB": metrics.mean_absolute_error(y_val, pred),
                              " % error": metrics.mean_squared_error(y_val, pred) / rang,
                              "Time":     total_time},
                              ignore_index=True)


results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['MSE', 'MAB'], vmin=0, vmax=100, color='#5fba7d')

print(results_ord)


print(y_train.max())
print(y_train.min())
print(y_val[3])
print(tree_classifiers['Random Forest'].predict(x_val)[3])


