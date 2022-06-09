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

class_weights = compute_class_weight (class_weight='balanced', classes=np.unique (y_train),y=y_train)
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
for i in range(114):
    idx2.append(i);

ytrain=[]
for j in y_train:
    ytrain.append(j);
#%%
#creation of the 3-fold splits of the cross validation
h=1;
ytrain= np.array(ytrain);
skfold2 = StratifiedKFold(n_splits=6,shuffle=True, random_state=0)
#, shuffle=True, random_state=1



fold_var = 1

working_dir='C:/Users/aurel'



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
optimizers=[tf.keras.optimizers.SGD(learning_rate=0.01),tf.keras.optimizers.SGD(learning_rate=0.001),tf.keras.optimizers.Adam(learning_rate=0.01),tf.keras.optimizers.SGD(learning_rate=0.001)]

#weight initialization by default is glorot uniform quite good for the weights of a perceptron with a sigmoid activation function.

for optim in optimizers:
    
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    VALIDATION_BAL_ACC=[]
    
    index = skfold2.split(x_train, ytrain);
    
    for train2_index, val_index in index:
         
         
        x_train2, x_val= x_train[train2_index], x_train[val_index]
        y_train2, y_val= ytrain[train2_index], ytrain[val_index]
    
    
        model = keras.Sequential()
        model.add(tf.keras.Input(shape=(11,)))
        model.add(keras.layers.Dense(units=1,
             activation='sigmoid'));
    
        #weight_metric=keras.metrics.Precision(name='precision')
        model.compile(optimizer=optim,loss='binary_focal_crossentropy', metrics=metric)
        
        
        
        filepath = os.path.join(working_dir, 'models', file_name(fold_var))
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, 
        				monitor='val_loss', verbose=1, 
        				save_best_only=True, mode='min')
        
        earlystop = EarlyStopping(monitor ="val_loss", min_delta = 0,patience = 10, verbose = 1,mode='min',restore_best_weights = True)
        callback = [checkpoint, earlystop]
        
        
        losses = model.fit(x_train2, y_train2,
         
                           validation_data=(x_val, y_val),
                            
                           # it will use 'batch_size' number
                           # of examples per example
                           batch_size=102,epochs=4000,callbacks=[callback],class_weight=class_weights);
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
        fold_var += 1	
        

    print('4-fold Cross-Validation accuracy',VALIDATION_ACCURACY)
    print('4-fold Cross-Validation mean accuracy',np.mean(VALIDATION_ACCURACY))
    print('4-fold Cross-Validation balanced accuracy',VALIDATION_BAL_ACC)
    print('4-fold Cross-Validation mean balanced accuracy',np.mean(VALIDATION_BAL_ACC))
    val_accuracy_values.append(VALIDATION_ACCURACY)
    val_mean_accuracy_values.append(np.mean(VALIDATION_ACCURACY))
    val_loss_values.append(VALIDATION_LOSS)
    val_bal_accuracy_values.append(VALIDATION_BAL_ACC)
    val_mean_bal_accuracy_values.append(np.mean(VALIDATION_BAL_ACC))



print('4-fold Cross-Validation mean accuracy', val_mean_accuracy_values);
print('4-fold Cross-Validation balanced accuracy', val_mean_bal_accuracy_values);


idx=argmax(val_mean_bal_accuracy_values);
optimal_optimizer=optimizers[idx];

model_opti = keras.Sequential()
model_opti.add(tf.keras.Input(shape=(11,)))
model_opti.add(keras.layers.Dense(units=1,
     activation='sigmoid'));
model_opti.compile(optimizer=optimal_optimizer,loss='binary_focal_crossentropy', metrics=metric)

earlystop = EarlyStopping(monitor ="val_loss", min_delta = 0,patience = 5, verbose = 1,mode='min',restore_best_weights = True)
callback = [earlystop]

losses = model_opti.fit(x_train, ytrain,
 
                   validation_data=(x_test, y_test),
                    
                   # it will use 'batch_size' number
                   # of examples per example
                   batch_size=86,epochs=4000,callbacks=[callback],class_weight=class_weights);

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
