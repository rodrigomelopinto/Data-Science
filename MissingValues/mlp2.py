from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.neural_network import MLPClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score, log_loss

file_tag = 'drought'
filename = 'dataWeek3/drought'
target = 'class'

train: DataFrame = read_csv(f'{filename}_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'{filename}_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

lr_type = ['constant', 'invscaling', 'adaptive']
max_iter = [100, 300, 500, 750, 1000, 2500, 5000]
learning_rate = [.1, .5, .9]
best = ('', 0, 0)
last_best = 0
best_model = None

'''cols = len(lr_type)
figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
for k in range(len(lr_type)):
    d = lr_type[k]
    values = {}
    for lr in learning_rate:
        yvalues = []
        for n in max_iter:
            mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                learning_rate_init=lr, max_iter=n, verbose=False)
            mlp.fit(trnX, trnY)
            prdY = mlp.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (d, lr, n)
                last_best = yvalues[-1]
                best_model = mlp
        values[lr] = yvalues
    multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
                           xlabel='mx iter', ylabel='accuracy', percentage=True)
savefig(f'images5th/{file_tag}_mlp_study.png')
show()
print(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')'''

'''clf = MLPClassifier(activation='logistic', solver='sgd', learning_rate='constant',
                                learning_rate_init=0.9, max_iter=300, verbose=False)
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'images5th/{file_tag}_mlp_constant_0.9_300.png')
show()'''

'''def plot_overfitting_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig(f'images5th/drought_overfitting_{name}.png')'''

def plot_loss_study(xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Loss {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig(f'images5th/drought_loss_{name}.png')


lr_type = 'adaptive'
lr = 0.1
eval_metric = log_loss
y_tst_values = []
y_trn_values = []
for n in max_iter:
    mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=lr_type, learning_rate_init=lr, max_iter=n, verbose=False)
    mlp.fit(trnX, trnY)
    prd_tst_Y = mlp.loss_
    prd_trn_Y = mlp.loss_
    y_tst_values.append(prd_tst_Y)
    y_trn_values.append(prd_trn_Y)
plot_loss_study(max_iter, y_trn_values, y_tst_values, name=f'NN_lr_type={lr_type}_lr={lr}', xlabel='nr max_iter', ylabel=str(eval_metric))