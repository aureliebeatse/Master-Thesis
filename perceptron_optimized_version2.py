# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:58:57 2022

@author: aurel
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
import scipy.stats
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from numpy import argmax
from sklearn_pandas import DataFrameMapper 
from tensorflow.keras import regularizers

#packages for the optimization of the parameters of the RandomForest
from sklearn import decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
import keras_tuner as kt
from tensorflow import keras
from keras_tuner import RandomSearch
from keras_tuner import Objective
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.utils import compute_class_weight

import traceback
import contextlib
import os




tp=keras.metrics.TruePositives(name='tp')
tn=keras.metrics.TrueNegatives(name='tn')
fp=keras.metrics.FalsePositives(name='fp')
fn=keras.metrics.FalseNegatives(name='fn')

def file_name(fold_var):
    return ('model'+str(fold_var))


def balanced_accuracy(y_true, y_pred):
    
    tp.reset_state()
    tn.reset_state()
    fp.reset_state()
    fn.reset_state()
    
    tp.update_state(y_true, y_pred);
    TP=tp.result()
    
    tn.update_state(y_true, y_pred);
    TN=tn.result()
    
    fp.update_state(y_true, y_pred);
    FP=fp.result()
    
    fn.update_state(y_true, y_pred);
    FN=fn.result()
    
    
    sensibi=sensibility(TP,FN)
    specifi=specificity(TN,FP)
    balanced_accuracy=bal_acc_calcul(sensibi,specifi);
  
    return balanced_accuracy;



@tf.function
def sensibility(true_pos, false_neg):
    
    sensi= true_pos/(true_pos+false_neg)    
    return sensi

@tf.function
def specificity(true_neg, false_pos):
    
    speci=true_neg/(true_neg+false_pos)
    
    return speci



@tf.function
def bal_acc_calcul(sensibi,specifi):
    
    
    bal_acc=(sensibi+specifi)/2;
    
    return bal_acc

#%%


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
#%%


df=pd.read_excel('C:/Users/aurel/OneDrive/Bureau/HAA_Dataset_final_2_modified.xlsx');
etiquette_categorie = dict( zip (df.Etiquette.unique(), df.Categorie.unique()));
print(etiquette_categorie); 
#%%

x = df [['Weight_Kg', 'Size_cm', 'Age_(yr)', 'Sex', 'Ascitis', 'Encephalopathy','INR_D0','Bilirubine_D0','Creatinine_D0', 'albumine_D0','WBC']];
y = df['Etiquette'];
a_u_c=[];
ci=[];

#to add sample weight

# sample_weights=np.ones(shape=(len(y)))
# sample_weights[x['Sex']==0] = 4/3
# sample_weights[x['Sex']==1] = 4/5

#standardisation for better performance


#ou normalisation par le range pour être dans l'interval 0-1
#x_norm=preprocessing.normalize(x, axis=0)

fold_var = 1
validation_results=[]
true_pos=keras.metrics.TruePositives()
true_neg=keras.metrics.TrueNegatives()
false_pos=keras.metrics.FalsePositives()
false_neg=keras.metrics.FalseNegatives()

#%%

#train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True,random_state=0,stratify=y);



cols_standardize = ['Weight_Kg', 'Size_cm', 'Age_(yr)', 'INR_D0', 'Bilirubine_D0','Creatinine_D0', 'albumine_D0', 'WBC']
cols_leave = ['Sex', 'Ascitis', 'Encephalopathy']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(x_train).astype('float32')
x_test = x_mapper.transform(x_test).astype('float32')

#%%

class_weights = compute_class_weight (class_weight='balanced', classes=np.unique(y_train),y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))
# weight_for_0=0.5
# weight_for_1=2
# class_weights={0: weight_for_0, 1: weight_for_1}
# weight = {0: weight_for_0, 1: weight_for_1}
with open('datasavingPerceptron.txt', mode='a') as file:
     file.write(str(class_weights))
     
input_shape = [x_train.shape[1]];
print('input shape of the perceptron',input_shape);

#%%
#reindex of y_train to correspond to index of x_train
idx2 = []
i=0;
for i in range(121):
    idx2.append(i);

ytrain=[]
for j in y_train:
    ytrain.append(j);
#%%
#creation of the 6-fold splits of the cross validation
h=1;
ytrain= np.array(ytrain);
skfold2 = StratifiedKFold(n_splits=6,shuffle=True, random_state=0)
#, shuffle=True, random_state=1



#fold_var = 1

#working_dir='C:/Users/aurel'



metric=[true_pos,true_neg, false_pos,false_neg, 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'), balanced_accuracy]

val_accuracy_values=[]
val_mean_accuracy_values=[]
val_loss_values=[]
val_bal_accuracy_values=[]
val_mean_bal_accuracy_values=[]
optimizers=[tf.keras.optimizers.Adam(learning_rate=0.01),tf.keras.optimizers.Adam(learning_rate=0.001)]
landas=[0,0.001,0.01] #change to 0 (test the 2)
#weight initialization by default is glorot uniform quite good for the weights of a perceptron with a sigmoid activation function.
#%%
for optim in optimizers:
    
    for landa in landas:
    
        VALIDATION_ACCURACY = []
        VALIDATION_LOSS = []
        VALIDATION_BAL_ACC=[]
        
        index = skfold2.split(x_train, ytrain);
        
        for train2_index, val_index in index:
             
             
            x_train2, x_val= x_train[train2_index], x_train[val_index]
            y_train2, y_val= ytrain[train2_index], ytrain[val_index]
        
        
            model = keras.Sequential()
            model.add(tf.keras.Input(shape=(11,)))
            model.add(keras.layers.Dense(units=1, kernel_regularizer=regularizers.L2(landa),
                 activation='sigmoid'));
        
            #weight_metric=keras.metrics.Precision(name='precision')
            model.compile(optimizer=optim,loss='binary_focal_crossentropy', metrics=metric)
            
            #choose binary focal cross entropy because it calculates the difference between the probability and the true label
            
            #filepath = os.path.join(working_dir, 'models', file_name(fold_var))
            # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, 
            # 				monitor='val_loss', verbose=1, 
            # 				save_best_only=True, mode='min')
            
            earlystop = EarlyStopping(monitor ="val_loss", min_delta = 0,patience = 15, verbose = 1,mode='min',restore_best_weights = True)
            callback = [earlystop]
            
            
            losses = model.fit(x_train2, y_train2,
             
                               validation_data=(x_val, y_val),
                                
                               # it will use 'batch_size' number
                               # of examples per example
                               batch_size=102,epochs=10000,callbacks=[callback],class_weight=class_weights);
            #sample_weight=sample_weights (ne pas oublier de le rajouter dans la fonction.fit)
            
            loss_df = pd.DataFrame(losses.history)
             
            # history stores the loss/val
            # loss in each epoch
             
            # loss_df is a dataframe which
            # contains the losses so we can
            # plot it to visualize our model training
            loss_df.loc[:,['loss','val_loss']].plot()
            
            loss_df.loc[:,['accuracy','val_accuracy']].plot()
            
            
            results = model.evaluate(x_val,y_val)
            results = dict(zip(model.metrics_names,results))
            
            VALIDATION_ACCURACY.append(results['accuracy'])
            VALIDATION_LOSS.append(results['loss'])
            VALIDATION_BAL_ACC.append(results['balanced_accuracy'])
            
            
            tf.keras.backend.clear_session()
            	
            
    
        print('6-fold Cross-Validation accuracy',VALIDATION_ACCURACY)
        print('6-fold Cross-Validation mean accuracy',np.mean(VALIDATION_ACCURACY))
        print('6-fold Cross-Validation balanced accuracy',VALIDATION_BAL_ACC)
        print('6-fold Cross-Validation mean balanced accuracy',np.mean(VALIDATION_BAL_ACC))
        val_accuracy_values.append(VALIDATION_ACCURACY)
        val_mean_accuracy_values.append(np.mean(VALIDATION_ACCURACY))
        val_loss_values.append(VALIDATION_LOSS)
        val_bal_accuracy_values.append(VALIDATION_BAL_ACC)
        val_mean_bal_accuracy_values.append(np.mean(VALIDATION_BAL_ACC))



print('6-fold Cross-Validation mean accuracy', val_mean_accuracy_values);
print('6-fold Cross-Validation balanced accuracy', val_mean_bal_accuracy_values);

#%%
tf.keras.backend.clear_session()

np.random.seed(1234)
tf.random.set_seed(2)

# idx=argmax(val_mean_bal_accuracy_values);
optimal_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

model_opti = keras.Sequential()
model_opti.add(tf.keras.Input(shape=(11,)))
model_opti.add(keras.layers.Dense(units=1,kernel_regularizer=regularizers.L2(0.001),
     activation='sigmoid'));
model_opti.compile(optimizer=optimal_optimizer,loss='binary_focal_crossentropy', metrics=metric)

earlystop = EarlyStopping(monitor ="val_loss", min_delta = 0,patience = 50, verbose = 1,mode='min',restore_best_weights = True)
callback = [earlystop]


index = skfold2.split(x_train, ytrain);

for trainopti_index, valopti_index in index:
    x_train_opti, x_val_opti= x_train[trainopti_index], x_train[valopti_index]
    y_train_opti, y_val_opti= ytrain[trainopti_index], ytrain[valopti_index]

    losses = model_opti.fit(x_train_opti, y_train_opti,
 
                   validation_data=(x_val_opti, y_val_opti),
                    
                   # it will use 'batch_size' number
                   # of examples per example
                   batch_size=102,epochs=4000,callbacks=[callback],class_weight=class_weights);
    loss_df = pd.DataFrame(losses.history)
         
        # history stores the loss/val
        # loss in each epoch
         
        # loss_df is a dataframe which
        # contains the losses so we can
        # plot it to visualize our model training
    loss_df.loc[:,['loss','val_loss']].plot()
        
    loss_df.loc[:,['accuracy','val_accuracy']].plot()
#on peut ajouter sample_weight si on veut aussi dans .fit ne pas oublier !
#%%
metric_predicted=model_opti.evaluate(x_test, y_test);

print('Best model Test metrics', metric_predicted);

#%%

y_pred=model_opti.predict(x_test)
print('model metrics', model_opti.metrics_names);

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)


#%%

# Evaluating the model

print('\nConfusion_Matrix                       : ');
plt.figure();
cm = confusion_matrix(y_test, y_pred);
cm_display = ConfusionMatrixDisplay(cm).plot();
plt.show();
#précision du modèle


    
#%%
alpha = .95

y_true = y_test
y_true=y_true.to_numpy()
y_pred=y_pred[:,0]

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


#%%
display = metrics.RocCurveDisplay(fpr=fpr_keras, tpr=tpr_keras, roc_auc=auc_keras);
    
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8));
display.plot(ax=ax1, color="red", label="Standardization_and_no_PCA");


ax=plt.gca()
col_labels=['AUC','AUC CI (95%)']
row_labels=['Standardization and no PCA']
table_vals=[[auc_,ci_]

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
#%%
display = metrics.RocCurveDisplay(fpr=fpr_keras, tpr=tpr_keras, roc_auc=auc_keras);
    
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8));
display.plot(ax=ax1, color="red", label="Standardization_and_no_PCA");


ax=plt.gca()
col_labels=['AUC','AUC CI (95%)']
row_labels=['Standardization and no PCA']
table_vals=[[auc_,ci_]]

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