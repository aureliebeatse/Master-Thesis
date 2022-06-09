# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:06:57 2022

@author: aurel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import torch # For building the networks 
import torchtuples as tt # Some useful functions
from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv
from torchtuples.practical import  _accuracy
from sksurv.metrics import cumulative_dynamic_auc
from keras.callbacks import History
import statistics
from keras import backend as K 
# We also set some seeds to make this reproducable.
# Note that on gpu, there is still some randomness.
np.random.seed(1234)
_ = torch.manual_seed(123)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


#%%

df=pd.read_excel('C:/Users/aurel/OneDrive/Bureau/HAA_Dataset_final_2_survival_new.xlsx');
etiquette_categorie = dict( zip (df.Etiquette.unique(), df.Categorie.unique()));
print(etiquette_categorie); 
y = df['Etiquette'];

#%%
data= df [['time','Status','Weight_Kg', 'Size_cm', 'Age_(yr)', 'Sex', 'Ascitis', 'Encephalopathy','INR_D0','Bilirubine_D0','Creatinine_D0', 'albumine_D0','WBC']];

print(data.head())


# on cree la base de donnee train/validation/test
df_train, df_test, ytrain, y_test = train_test_split(data, y, test_size=0.20, shuffle=True,random_state=0,stratify=y);
# df_test = data.sample(frac=0.2)
# df_train = data.drop(df_test.index)
# df_val = df_train.sample(frac=0.2)
# df_train = df_train.drop(df_val.index)


print(df_train.head())
skfold2 = StratifiedKFold(n_splits=6,shuffle=True, random_state=0)

#index = skfold2.split(x_train, ytrain);


cols_standardize = ['Weight_Kg', 'Size_cm', 'Age_(yr)', 'INR_D0', 'Bilirubine_D0','Creatinine_D0', 'albumine_D0', 'WBC']
cols_leave = ['Sex', 'Ascitis', 'Encephalopathy']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)



df_train_standard = x_mapper.fit_transform(df_train).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

index = skfold2.split(df_train, ytrain);

#on fit sur les donnees train et on applique la transformee (donc la standardisation) sur train, val et test
num_durations = 10  #nombre de neurones de la couche de sortie
labtrans = LogisticHazard.label_transform(num_durations)
print(type(labtrans))
get_target = lambda df: (df['time'].values, df['Status'].values)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
#%%
#parameters of the model
#number_nodes=[[3],[4],[5],[10],[15],[20],[25]]
number_nodes=[5,6,10,11,17,18,19,20,21]
#num_nodes = [32, 32]
batch_norm = True
dropout_values=[0,0.1,0.2,0.3,0.4,0.5]
#dropout = 0.1

#parameters of the training

history=History()
#metrics = {'acc': _accuracy}

c_index_mean_vec=[]
c_index_global_vec=[]
integrated_brier_score_mean_vec=[]
brier_score_global_vec=[]
epochs_number_vec=[]

for num_nodes in number_nodes:

    for dropout in dropout_values:
    
        c_index_vec=[]
        integrated_brier_score_vec=[]
        epochs_number=[]
        
        index = skfold2.split(df_train, ytrain);
        
        for train2_index, val_index in index:
            np.random.seed(1234)
            _ = torch.manual_seed(123)
            import torch # For building the networks 
            import torchtuples as tt # Some useful functions
            from pycox.models import LogisticHazard
            x_train, x_val= df_train.iloc[train2_index,:], df_train.iloc[val_index,:]
            durations_val, events_val = get_target(x_val)
            durations_train, events_train = get_target(x_train)
            
        
        
        # labtrans = PMF.label_transform(num_durations)
        # labtrans = DeepHitSingle.label_transform(num_durations)
        
            y_train = labtrans.fit_transform(*get_target(x_train))
            y_val = labtrans.transform(*get_target(x_val))
            
            x_train=x_mapper.transform(x_train).astype('float32')
            x_val = x_mapper.transform(x_val).astype('float32')
            
            train = (x_train, y_train)
            val = (x_val, y_val)
        
        
        
        
        
            print('Time point for the survival curve',labtrans.cuts) #print the cut points
        
        
        
        
            print('Index of survival time points and event status',y_train)
        
        
            print('Survival time of the train dataset',labtrans.cuts[y_train[0]])
        
        
        #Make neural network
        
            in_features = x_train.shape[1]
            
            out_features = labtrans.out_features
            
            
            net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
        
        
            #definign the model
            model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
            batch_size = 34
            epochs = 1000
            callbacks = [tt.cb.EarlyStopping(patience=15)]
        
        
        #entrainement
        
            log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
        
        
        #on plot la loss
            log.to_pandas()[['train_loss', 'val_loss']].plot()
            plt.xlabel('epoch')
            _ = plt.ylabel('loss')
            number_of_epochs=len(log.to_pandas()['val_loss'])
            epochs_number.append(number_of_epochs)
        
        #on plot l'accuracy
        # log.to_pandas()[['train_acc', 'val_acc']].plot()
        # plt.xlabel('epoch')
        # _ = plt.ylabel('accuracy')
        
        
            print('Minimum validation loss',log.to_pandas().val_loss.min())
        
        # we verify that we choose the best model with the minimum validation loss
            print('Score of the saved model on validation dataset',model.score_in_batches(val))
        
        
        #print('Score of the saved model on the training dataset',model.score_in_batches(train))
        
        #evaluation on the validation data with some criteria
        
            surv = model.predict_surv_df(x_val);
            
        
            ev = EvalSurv(surv, durations_val, events_val, censor_surv='km')
            c_index_td=ev.concordance_td('antolini')
            c_index_vec.append(c_index_td);
        
            print('Time Dependant Concordance Index',c_index_td)
        
        
            time_grid = np.linspace(durations_val.min(), durations_val.max(), 100)
            int_brier_score=ev.integrated_brier_score(time_grid)
            print('Integrated Brier Score', int_brier_score)
            integrated_brier_score_vec.append(int_brier_score)
            
            del model
            del net
            K.clear_session()
            
        
        #     transposed_surv= surv.transpose()
        #     surv_array=transposed_surv.to_numpy();
            
        # #%%
        #     events_train_binary=[]
        #     for event in events_train:
        #         if event==0:
        #             events_train_binary.append(False);
        #         if event==1:
        #             events_train_binary.append(True);
            
        #     events_val_binary=[]
        #     for event in events_val:
        #         if event==0:
        #             events_val_binary.append(False);
        #         if event==1:
        #             events_val_binary.append(True);
        # #%%
        #     #survival_train=(events_train, durations_train)
        #     temp = [(e1,t1) for e1,t1 in zip(events_train_binary, durations_train)]
        #     survival_train = np.array(temp, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        # #%%
        #     temp2= [(e2,t2) for e2,t2 in zip(events_val_binary, durations_val)]
        #     survival_val= np.array(temp2,  dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        # #%%
        #     times=np.unique(durations_val)
        #     times=times[0:13]
        # #%%
        #     #"dynamic_auc=cumulative_dynamic_auc(survival_train, survival_val, estimate=surv_array,times=times);
            
            #print('Cumulative_Dynamic_AUC', dynamic_auc)
        
        c_index_global_vec.append(c_index_vec)
        brier_score_global_vec.append(integrated_brier_score_vec)
        print('Time dependant c-index over 4-fold cross-validation',c_index_vec)
        print('Mean td c-index over the 4-fold', statistics.mean(c_index_vec))
        
        c_index_mean_vec.append(statistics.mean(c_index_vec))
        print('Integrated Brier Score over 4-fold cross-validation', integrated_brier_score_vec)
        print('Mean Integrated brier Score over the 4-fold', statistics.mean(integrated_brier_score_vec))
        
        integrated_brier_score_mean_vec.append(statistics.mean(integrated_brier_score_vec))
        epochs_number_vec.append(epochs_number)




#%%



#%%
np.random.seed(1234)
_ = torch.manual_seed(123)
import torch # For building the networks 
import torchtuples as tt # Some useful functions
from pycox.models import LogisticHazard
n_node_opti=5
drop=0.1
in_features = x_train.shape[1]
out_features = labtrans.out_features
net_opti = tt.practical.MLPVanilla(in_features, n_node_opti, out_features, batch_norm, drop)
model_opti = LogisticHazard(net_opti, tt.optim.Adam(0.01), duration_index=labtrans.cuts)


skfold_opti = StratifiedKFold(n_splits=6,shuffle=True, random_state=0) #random state de 42
index2 = skfold_opti.split(df_train, ytrain);


for trainopti_index, valopti_index in index2:
    x_train_opti, x_val_opti= df_train.iloc[trainopti_index,:], df_train.iloc[valopti_index,:]
    # durations_val, events_val = get_target(x_val)
    # durations_train, events_train = get_target(x_train)

    y_train_opti = labtrans.fit_transform(*get_target(x_train_opti))
    y_val_opti = labtrans.transform(*get_target(x_val_opti))
    
    x_train_opti=x_mapper.transform(x_train_opti).astype('float32')
    x_val_opti = x_mapper.transform(x_val_opti).astype('float32')
    
    train_opti= (x_train_opti, y_train_opti)
    val_opti = (x_val_opti, y_val_opti)



    batch_size = 34
    epochs = 1000
    callbacks = [tt.cb.EarlyStopping(patience=15)]
    
    
    
    log = model_opti.fit(x_train_opti, y_train_opti, batch_size, epochs, callbacks, val_data=val_opti)
    log.to_pandas()[['train_loss', 'val_loss']].plot()
    plt.xlabel('epoch')
    _ = plt.ylabel('loss')
# number_of_epochs=len(log.to_pandas()['val_loss'])
# epochs_number.append(number_of_epochs)


print('Score of the saved model on validation dataset',model_opti.score_in_batches(val_opti))

#prediction with the function predict_surv_df to obtain the probability estimate of survival of each timepoint in a dataframe
#predictions done on the test set
surv_opti = model_opti.predict_surv_df(x_test);


#plot of the survival curve estimation of the 5 first patients
#they are step functions

surv_opti.iloc[:, 5:9].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')


#interpolation of the survival estimations

surv_opti_interp = model_opti.interpolate(10).predict_surv_df(x_test);


#plot the survival curve estimations which have been interpolated
surv_opti_interp.iloc[:, 5:9].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')



#evaluation of the model with some criteria
ev = EvalSurv(surv_opti, durations_test, events_test, censor_surv='km')


#concordance index
print('Concordance index for evaluation of the test dataset',ev.concordance_td('antolini'))


#Brier Score
plt.figure()
time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
ev.brier_score(time_grid).plot()
plt.ylabel('Brier score')
_ = plt.xlabel('Time')


# IPCW negative binomial log-likelihood
# ev.nbll(time_grid).plot()
# plt.ylabel('NBLL')
# _ = plt.xlabel('Time')


#integrated score by numerical integration over a timegrid of the two above scores

print('Integrated Brier Score',ev.integrated_brier_score(time_grid))

#print('Integrated Negative Binomial Log Likelihood',ev.integrated_nbll(time_grid))
#%%
surv_opti = model_opti.predict_surv_df(x_train);
ev = EvalSurv(surv_opti, durations_train, events_train, censor_surv='km')
print('Concordance index for evaluation of the test dataset',ev.concordance_td('antolini'))

#%%

transposed_surv= surv_opti.transpose()
surv_array=transposed_surv.to_numpy();
surv_array=np.delete(surv_array,0,axis=1)

#%%
events_train_binary=[]
for event in events_train:
    if event==0:
        events_train_binary.append(False);
    if event==1:
        events_train_binary.append(True);

events_val_binary=[]
for event in events_test:
    if event==0:
        events_val_binary.append(False);
    if event==1:
        events_val_binary.append(True);
#%%
#survival_train=(events_train, durations_train)
temp = [(e1,t1) for e1,t1 in zip(events_train_binary, durations_train)]
survival_train = np.array(temp, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
#%%
durations_val2=[]
for d in durations_test:
    
    if d==120:
        durations_val2.append(121)
    else:
        durations_val2.append(d)
       
durations_val2=np.array(durations_val2)
#%%
temp2= [(e2,t2) for e2,t2 in zip(events_val_binary, durations_val2)]
survival_val= np.array(temp2,  dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
#%%
times=labtrans.cuts
times=np.delete(times,0)
#%%
dynamic_auc=cumulative_dynamic_auc(survival_train, survival_val, estimate=surv_array,times=times);

print('Cumulative_Dynamic_AUC', dynamic_auc)
# np.random.seed(1234)
# _ = torch.manual_seed(123)
# import torch # For building the networks 
# import torchtuples as tt # Some useful functions
# from pycox.models import LogisticHazard
# n_node_opti=5
# drop=0.2
# in_features = x_train.shape[1]
# out_features = labtrans.out_features
# net_opti = tt.practical.MLPVanilla(in_features, n_node_opti, out_features, batch_norm, drop)
# model_opti = LogisticHazard(net_opti, tt.optim.Adam(0.01), duration_index=labtrans.cuts)




# x_train_opti, x_val_opti = train_test_split(df_train, test_size=0.10, shuffle=True,random_state=0,stratify=ytrain);


    
#     # durations_val, events_val = get_target(x_val)
#     # durations_train, events_train = get_target(x_train)

# y_train_opti = labtrans.fit_transform(*get_target(x_train_opti))
# y_val_opti = labtrans.transform(*get_target(x_val_opti))

# x_train_opti=x_mapper.transform(x_train_opti).astype('float32')
# x_val_opti = x_mapper.transform(x_val_opti).astype('float32')

# train_opti= (x_train_opti, y_train_opti)
# val_opti = (x_val_opti, y_val_opti)



# batch_size = 92
# epochs = 1000
# callbacks = [tt.cb.EarlyStopping(patience=15)]
    
    
    
# log = model_opti.fit(x_train_opti, y_train_opti, batch_size, epochs, callbacks, val_data=val_opti)
# log.to_pandas()[['train_loss', 'val_loss']].plot()
# plt.xlabel('epoch')
# _ = plt.ylabel('loss')
# # number_of_epochs=len(log.to_pandas()['val_loss'])
# # epochs_number.append(number_of_epochs)

# print('Score of the saved model on validation dataset',model_opti.score_in_batches(val_opti))

# #prediction with the function predict_surv_df to obtain the probability estimate of survival of each timepoint in a dataframe
# #predictions done on the test set
# surv_opti = model_opti.predict_surv_df(x_test);


# #plot of the survival curve estimation of the 5 first patients
# #they are step functions

# surv_opti.iloc[:, :5].plot(drawstyle='steps-post')
# plt.ylabel('S(t | x)')
# _ = plt.xlabel('Time')


# #interpolation of the survival estimations

# surv_opti_interp = model_opti.interpolate(10).predict_surv_df(x_test);


# #plot the survival curve estimations which have been interpolated
# surv_opti_interp.iloc[:, :5].plot(drawstyle='steps-post')
# plt.ylabel('S(t | x)')
# _ = plt.xlabel('Time')



# #evaluation of the model with some criteria
# ev = EvalSurv(surv_opti, durations_test, events_test, censor_surv='km')


# #concordance index
# print('Concordance index for evaluation of the test dataset',ev.concordance_td('antolini'))


# #Brier Score
# plt.figure()
# time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
# ev.brier_score(time_grid).plot()
# plt.ylabel('Brier score')
# _ = plt.xlabel('Time')


# # IPCW negative binomial log-likelihood
# # ev.nbll(time_grid).plot()
# # plt.ylabel('NBLL')
# # _ = plt.xlabel('Time')


# #integrated score by numerical integration over a timegrid of the two above scores

# print('Integrated Brier Score',ev.integrated_brier_score(time_grid))

# #print('Integrated Negative Binomial Log Likelihood',ev.integrated_nbll(time_grid))