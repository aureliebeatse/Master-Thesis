# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:23:48 2022

@author: aurel
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
import scipy.stats
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn_pandas import DataFrameMapper 

#packages for the optimization of the parameters of the LR
from sklearn import decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold



# AUC comparison adapted from
# https://github.com/Netflix/vmaf/µ

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    print("k",k)
    
    

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    print("prediction_sorted_transposed", predictions_sorted_transposed)
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov



#-----------------------------------------------------------------------------------------------------




def plotting_roc_curves(fpr, tpr, y_test, survival_predictions_proba):
    
    # other way of plotting ROC curves (from estimator= modele_regLog= regression logistique)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    roc_auc = metrics.auc(fpr, tpr);
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc);
    display.plot(ax=ax1);
    
    
    
    # other way of plotting ROC curves (from predictions= survival_prediction)
    
    RocCurveDisplay.from_predictions(y_test, survival_predictions_proba[:,1], pos_label=1, ax=ax2);
    plt.show();
    
    
    
    
    return(display)

#%%

with open('datasaving.txt', mode='w+') as file:
    file.write("Execution code RL: Dataset2_modified, train/test split=0.75/0.25, balanced class_weight\n")

#%%

df=pd.read_excel('C:/Users/aurel/OneDrive/Bureau/HAA_Dataset_final_2_modified.xlsx');

etiquette_categorie = dict( zip (df.Etiquette.unique(), df.Categorie.unique()));
print(etiquette_categorie);                   


#%%
#variables creation
x = df [['Weight_Kg', 'Size_cm', 'Age_(yr)', 'Sex', 'Ascitis', 'Encephalopathy','INR_D0','Bilirubine_D0','Creatinine_D0', 'albumine_D0','WBC']];
y = df['Etiquette'];
a_u_c=[];
ci=[];

pca=decomposition.PCA()

#cov=p_c_a.get_covariance();
#print('covariance des features de x',cov);
#print('explained variance ratio',p_c_a.explained_variance_ratio_);
#print('components',p_c_a.components_);

    
#fractionner dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True,random_state=0,stratify=y);

cols_standardize = ['Weight_Kg', 'Size_cm', 'Age_(yr)', 'INR_D0', 'Bilirubine_D0','Creatinine_D0', 'albumine_D0', 'WBC']
cols_leave = ['Sex', 'Ascitis', 'Encephalopathy']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(x_train).astype('float32')
x_test = x_mapper.transform(x_test).astype('float32')

#%%
idx2 = []
i=0;
for i in range(114):
    idx2.append(i);

ytrain=[]
for j in y_train:
    ytrain.append(j);
    
#idx2=train_index;
#x_train= pd.DataFrame(x_train,index=idx2);
#%%
ytrain= np.array(ytrain);
skfold2 = StratifiedKFold(n_splits=6,shuffle=True, random_state=0)
#, shuffle=True, random_state=1
index = skfold2.split(x_train, ytrain);

#%%
#juste pour avoir la valeur des index
for train2_index, val_index in index:
     
      #x_train2, x_val= x_train.iloc[train2_index,:], x_train.iloc[val_index,:]
      x_train2, x_val= x_train[train2_index], x_train[val_index]
      y_train2, y_val= ytrain[train2_index], ytrain[val_index]
#%%
#instanciation du modèle

modele_regLog = linear_model.LogisticRegression(class_weight='balanced',max_iter=10000, n_jobs=-1, random_state=0);
pipe = Pipeline(steps=[
    ('pca', pca),
    ('logistic_Reg', modele_regLog)])

pca_components=list(range(1,x.shape[1]+1,1));
#{'pca__n_components':[1,2,3,4,5,6,7,8,9,10,11],'logistic_Reg__penalty': ['l1'], 'logistic_Reg__solver': ['liblinear','saga'], 'logistic_Reg__C':[0.5,1,1.5,2,2.5]},
param_grid = [{'pca__n_components':[1,2,3,4,5,6,7,8,9,10,11],'logistic_Reg__penalty':['l2'],'logistic_Reg__solver': ['newton-cg','lbfgs','liblinear','sag', 'saga'], 'logistic_Reg__C':[1,10,100,1000]}];
scores=['balanced_accuracy','precision_macro', 'recall_macro','f1_macro','roc_auc']
#%%

# parameters = [{'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
#               {'penalty':['none', 'elasticnet', 'l1', 'l2']},
#               {'C':[0.001, 0.01, 0.1, 1, 10, 100]}]



# grid_search = GridSearchCV(estimator = modele_regLog,  
#                            param_grid = parameters,
#                            scoring = 'balanced_accuracy',
#                            cv = 3,
#                            verbose=0);


#pipe = Pipeline(steps=[('std_slc', std_slc),
                   # ('pca', pca),
                   # ('logistic_Reg', modele_regLog)]);

#n_components = list(range(1,x.shape[1]+1,1));
#C = np.logspace(-4, 4, 50);
#penalty = ['l1', 'l2'];
#solver=['newton-cg', 'lbfgs', 'liblinear', 'saga'];
#parameters = dict(pca__n_components=n_components,
                  # logistic_Reg__C=C,
                  # logistic_Reg__penalty=penalty)#, logistic_Reg__solver=solver);
grid_search= GridSearchCV(pipe, param_grid, scoring='balanced_accuracy',n_jobs=-1,cv=skfold2.split(x_train, ytrain), return_train_score=True);
grid_search.fit(x_train, ytrain);

#%%

print('Best Penalty:', grid_search.best_estimator_.get_params()['logistic_Reg__penalty'])
print('Best Solver:', grid_search.best_estimator_.get_params()['logistic_Reg__solver'])
print('Best C:', grid_search.best_estimator_.get_params()['logistic_Reg__C'])
print('Best Number Of Components:', grid_search.best_estimator_.get_params()['pca__n_components'])

n=grid_search.best_estimator_.get_params()['pca__n_components']

with open('datasaving.txt', mode='a') as file:
    file.write("\nBest number of components for PCA:");
    file.write(str(grid_search.best_estimator_.get_params()['pca__n_components']))

#%%
LR_parameters=grid_search.best_estimator_.get_params()['logistic_Reg'];
best_parameters=LR_parameters.get_params();
print('Best Parameters of LR:', best_parameters)

with open('datasaving.txt', mode='a') as file:
    file.write("\nBest parameters of LR:\n");
    file.write(str(best_parameters))

best_model=modele_regLog.set_params(**best_parameters)
pca = decomposition.PCA(n_components=n);

x_train_pca=pca.fit_transform(x_train)
x_test_pca=pca.transform(x_test)
best_model.fit(x_train_pca,y_train);
#%%
#Use of cross_validation to evaluate the model
CV_Results= cross_val_score(best_model, x_train_pca ,y_train,scoring='balanced_accuracy', cv=skfold2.split(x_train_pca, ytrain), n_jobs=-1); #n_jobs=-1 means using all processors available
print('Cross-Validation Results', CV_Results);
print('Cross-Validation Mean Results', CV_Results.mean());
print('Cross-Validation Standard Deviation Results', CV_Results.std());
#dual=False, tol=0.0001, 
#C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

with open('datasaving.txt', mode='a') as file:
    file.write("\n6-fold Cross-validation Results:");
    file.write(str(CV_Results));
    file.write("\n6-fold Cross-validation Mean Results:");
    file.write(str(CV_Results.mean()));
    file.write("\n6-fold Cross-validation Standard Deviation Results:");
    file.write(str(CV_Results.std()));

#%%
#training

with open('datasaving.txt', mode='a') as file:
    file.write("\nCoeff Regression Logistique:");
    file.write(str(best_model.coef_))

#prédiction
survival_prediction=best_model.predict(x_test_pca);
survival_predictions_proba = best_model.predict_proba(x_test_pca);
print(etiquette_categorie[survival_prediction[0]], etiquette_categorie[y_test.iloc[0]]);

#%%
print('\nConfusion_Matrix                       : ');
plt.figure();
cm = confusion_matrix(y_test, survival_prediction);
cm_display = ConfusionMatrixDisplay(cm).plot();
plt.show();
#précision du modèle
acc = best_model.score(x_test_pca,y_test);
print('\nGlobal Accuracy                       : ');
print(acc*100);
with open('datasaving.txt', mode='a') as file:
    file.write("\nGlobal accuracy of the model:");
    file.write(str(acc*100))
print('\nClassification Report                       : ');
print(classification_report(y_test, survival_prediction));
with open('datasaving.txt', mode='a') as file:
    file.write("\nClassification Metrics Report: \n");
    file.write(str(classification_report(y_test, survival_prediction)));
balanced_accuracy=balanced_accuracy_score(y_test, survival_prediction, adjusted=False);
print('\nBalanced Accuracy                       : ');
print(balanced_accuracy);
with open('datasaving.txt', mode='a') as file:
    file.write("\nBalanced Accuracy:");
    file.write(str( balanced_accuracy));
print('\nJ Youdens statistics                       : ');
J_Youdens_score=balanced_accuracy_score(y_test, survival_prediction, adjusted= True);
print(J_Youdens_score);
with open('datasaving.txt', mode='a') as file:
    file.write("\nJ Youdens Statistics:");
    file.write(str(J_Youdens_score));


alpha = .95
y_pred = survival_predictions_proba[:,1]
y_true = y_test



auc_, auc_cov = delong_roc_variance(y_true, y_pred)
a_u_c.append(auc_);
with open('datasaving.txt', mode='a') as file:
    file.write("\nAUC:");
    file.write(str(auc_))


auc_std = np.sqrt(auc_cov)
lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)



ci_= stats.norm.ppf(
    lower_upper_q,
    loc=auc_,
    scale=auc_std)


ci_[ci_ > 1] = 1
ci.append(ci_);
with open('datasaving.txt', mode='a') as file:
    file.write("\nAUC Confidence Interval at 95%:");
    file.write(str(ci_));

print('AUC:', auc_)
print('AUC COV:', auc_cov)
print('95% AUC CI:', ci_)

#%%ROC curve and score

auc = roc_auc_score(y_test, survival_predictions_proba[:, 1]);
fpr, tpr, thresholds = roc_curve(y_test, survival_predictions_proba[:,1], pos_label=1);
display= plotting_roc_curves(fpr, tpr, y_test, survival_predictions_proba);


#%% 1 way of plotting ROC curve
#on calcule les false positive rate et true positive rate pour differentes valeurs de treshold
# with plt.style.context(('ggplot','seaborn')):
#     plt.figure(figsize=(8,6))
#     plt.scatter(fpr, tpr, c='blue')
#     plt.subplot(2,2,1)
#     plt.plot(fpr, tpr, label="Accuracy:%.2f AUC:%.2f" % (acc, auc), linewidth=2, c='red')
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate (recall)")
#     plt.title('ROC Curve')
#     plt.legend(loc='best');



#%%

file.close();
    
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8));
display.plot(ax=ax1, color="red", label="Standardization_and_PCA");


ax=plt.gca()
col_labels=['AUC','AUC CI (95%)']
row_labels=['Standardization and PCA']
table_vals=[[a_u_c[0],ci[0]]]

# the rectangle is where I want to place the table
the_table = plt.table(cellText=table_vals, cellLoc='center',
                  colWidths = [0.12]*3,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc="lower center", colLoc="center")
the_table.auto_set_font_size(False)
the_table.set_fontsize(8);
the_table.scale(1.5, 1.5); 

#plt.text(12,3.4,'Table Title',size=8)
plt.show();


