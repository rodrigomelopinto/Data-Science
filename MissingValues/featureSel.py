#%%
from pandas import read_csv
from pandas import DataFrame
from typing import Tuple
from matplotlib.pyplot import figure, title, savefig, show
from seaborn import heatmap
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types

file_tag = 'diabetic'
filename = 'dataWeek3/diabetic_over.csv'
data = read_csv(filename, na_values='')
'''
THRESHOLD = 0.9

def select_redundant(corr_mtx, threshold: float) -> Tuple[dict, DataFrame]:
    if corr_mtx.empty:
        return {}

    corr_mtx = abs(corr_mtx)
    vars_2drop = {}
    for el in corr_mtx.columns:
        el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= threshold]
        if len(el_corr) == 1:
            corr_mtx.drop(labels=el, axis=1, inplace=True)
            corr_mtx.drop(labels=el, axis=0, inplace=True)
        else:
            vars_2drop[el] = el_corr.index
    return vars_2drop, corr_mtx

drop, corr_mtx = select_redundant(data.corr(), THRESHOLD)
print(drop.keys())

if corr_mtx.empty:
    raise ValueError('Matrix is empty.')

figure(figsize=[10, 10])
heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
title('Filtered Correlation Analysis')
savefig(f'imagesWeek3/filtered_correlation_analysis_{THRESHOLD}.png')
show()

def drop_redundant(data: DataFrame, vars_2drop: dict) -> DataFrame:
    sel_2drop = []
    print(vars_2drop.keys())
    for key in vars_2drop.keys():
        if key not in sel_2drop:
            for r in vars_2drop[key]:
                if r != key and r not in sel_2drop:
                    sel_2drop.append(r)
    print('Variables to drop', sel_2drop)
    df = data.copy()
    for var in sel_2drop:
        df.drop(labels=var, axis=1, inplace=True)
    return df
df = drop_redundant(data, drop)'''

data = data.drop(['Unnamed: 0','examide', 'citoglipton', 'acetohexamide', 'glimepiride-pioglitazone'], axis=1)
data.to_csv(f'dataWeek3/{file_tag}_sel.csv', index=False)
# %%
