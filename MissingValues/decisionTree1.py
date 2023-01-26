#%%
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show, imread,imshow,axis,Axes
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from ds_charts import plot_evaluation_results_tern, multiple_line_chart, horizontal_bar_chart, plot_overfitting_study
from sklearn.metrics import accuracy_score
from subprocess import call
from sklearn import tree
from numpy import argsort, arange


file_tag = 'diabetic_sel'
filename = 'dataWeek3/diabetic_sel'
target = 'readmitted'

train: DataFrame = read_csv(f'{filename}_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'{filename}_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
max_depths = [2, 5, 10, 15, 20, 25]
criteria = ['entropy', 'gini']
best = ('',  0, 0.0)
last_best = 0
best_model = None

fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)
for k in range(len(criteria)):
    f = criteria[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for imp in min_impurity_decrease:
            tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
            tree.fit(trnX, trnY)
            prdY = tree.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (f, d, imp)
                last_best = yvalues[-1]
                best_model = tree

        values[d] = yvalues
    multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                           xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
savefig(f'imagesWeek4/{file_tag}_dt_study.png')
show()
print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

'''
file_tree = 'imagesWeek4/drought_best_tree.png'

#dot_data = export_graphviz(best_model, out_file='imagesWeek4/best_tree.dot', filled=True, rounded=True, special_characters=True)
# Convert to png

call(['dot', '-Tpng', 'imagesWeek4/drought_best_tree.dot', '-o', file_tree, '-Gdpi=28'])

figure(figsize = (14, 18))
imshow(imread(file_tree))
axis('off')
show()'''
'''
best_model = DecisionTreeClassifier(max_depth=10, criterion='entropy', min_impurity_decrease=0.00)
best_model.fit(trnX, trnY)
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results_tern(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'imagesWeek4/{file_tag}_dt_md=10_c=entr_mi=0.png')
show()'''



'''
variables = train.columns
importances = best_model.feature_importances_
indices = argsort(importances)[::-1]
elems = []
imp_values = []
for f in range(len(variables)):
    elems += [variables[indices[f]]]
    imp_values += [importances[indices[f]]]
    print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

figure()
horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
savefig(f'imagesWeek4/{file_tag}_dt_ranking.png')




imp = 0.0001
f = 'entropy'
eval_metric = accuracy_score
y_tst_values = []
y_trn_values = []
for d in max_depths:
    tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
    tree.fit(trnX, trnY)
    prdY = tree.predict(tstX)
    prd_tst_Y = tree.predict(tstX)
    prd_trn_Y = tree.predict(trnX)
    y_tst_values.append(eval_metric(tstY, prd_tst_Y))
    y_trn_values.append(eval_metric(trnY, prd_trn_Y))
figure()
plot_overfitting_study(max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}', xlabel='max_depth', ylabel=str(eval_metric))
savefig(f'imagesWeek4/{file_tag}_overfitting_DT=imp{imp}_{f}.png')'''
# %%
