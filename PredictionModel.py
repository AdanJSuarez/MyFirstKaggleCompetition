

#### Import 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from sklearn import svm

#### Load data
data_train = pd.read_csv('~/Documents/Dataset/Kaggle/SberbankRussianHousingMarket/trainclean.csv')
data_test = pd.read_csv('~/Documents/Dataset/Kaggle/SberbankRussianHousingMarket/testclean.csv')

#### Helper function

def rmsle(y_predicted, y_real):
    ''' y_predicted and y_real must to be pd.Series '''
    # Check if they are comparable
    print('RMSLE processing...')
    if len(y_predicted) != len(y_real):
        print('Different len between y_predicted and y_real')
        return
    val1 = []
    for i in range(len(y_predicted)):
        val2 = math.log(y_predicted[i] + 1) - math.log(y_real[i] + 1)
        val1.append(val2)
    e = ((1/len(y_predicted)) * (sum(val1))**2)**1/2
    return e

def data_split(data, percent=0.25):
    rng = round(percent*len(data))
    x_train = data[:-rng]
    x_test = data[-rng +1:]
    return x_train,x_test

# def data_split(data,percent=0.25):
#     X_train, X_test, y_train, y_test = train_test_split(data, data['Price/full_sq'], test_size=percent)
#     return X_train, X_test, y_train, y_test

def recti_price(data, rectificator):
    r = rectificator/len(data)
    rec = 1 - r
    for i in data.index:
        data.ix[i] = round(data.ix[i] * rec,2)
        rec -= r
    return data

def predictive_model(data_train, data_test):
    '''Predictive model'''
    mean = data_train['Price/full_sq'].describe()[1]
    std = data_train['Price/full_sq'].describe()[2]
    umbral_up = mean + 1*std
    umbral_down = mean - std
    var5 = pd.DataFrame()
    for i in set(data_train['sub_area']):
        print('Processed = ', round((i/len(set(data_train['sub_area'])))*100,2), end=" ")
        var1 = data_train.loc[data_train['sub_area'] == i]
        var2 = data_test.loc[data_test['sub_area'] == i]
        index = var2.id.apply(lambda x: int(x))
        # If not enough training data point use mean
        if len(var1) < 2:
            print('Applying mean')
            #mean = var1['Price/full_sq'].mean()
            var6 = pd.Series(mean, index=index)
            var5 = pd.concat([var5, var6])
            
        # If data_test has no data for this area
        elif len(var2) == 0:
            pass
        else:
            x_train = var1
            x_test = var2
            
            # Changing weirdos for mean
            weirdos_up = x_train.loc[x_train['Price/full_sq'] > umbral_up]
            if  len(weirdos_up) > 0:
                #mean = x_train['Price/full_sq'].mean()
                x_train.at[x_train['Price/full_sq'] > umbral_up,'Price/full_sq'] = mean
                print(end="\r")
            
            y_train = var1['Price/full_sq']
            
            # Droping unnecessary columns
            var3 = ['id','Price/full_sq', 'price_doc', 'sub_area', 'timestamp',
                    'hospital_beds_raion','cafe_sum_500_min_price_avg',
                    'cafe_sum_500_max_price_avg', 'cafe_avg_price_500']
            for j in var3:
                x_train = x_train.drop(j,1)
            var4 = ['id','sub_area','timestamp']
            for j in var4:
                x_test = x_test.drop(j,1)
                
            # Reduce features
            # x_train = x_train.T[:10].T
            # x_test = x_test.T[:10].T
            # Reduce features
            # rng1 = x_train.T[13:80].index
            # rng2 = x_train.T[149:283].index
            # #rng = [x for x in range(16,83)] + [x for x in range(152,286)]
            # for i in rng1:
            #     x_train = x_train.drop(i,1)
            #     x_test = x_test.drop(i,1)
            # for i in rng2:
            #     x_train = x_train.drop(i,1)
            #     x_test = x_test.drop(i,1)
                
            # Standarization            
            scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        
            # Regression model
            print('Model Processing...', end="\r")
            model = SVR(kernel='rbf', C=11.5, epsilon=1.5, gamma=2/x_train.shape[1])
            #model =  RandomForestClassifier(n_estimators=500, n_jobs=-1,random_state = 0)
            y = pd.Series(model.fit(x_train, y_train).predict(x_test), index=index)
                
            var5 = pd.concat([var5, y])
            
    for i in data_test.id:
        var5.ix[int(i)] = round(var5.ix[int(i)] * float(data_test.loc[data_test.id == i, 'full_sq']),2)
    return var5


#### End helper function

print('#### Start computing ####')


### Make predition by areas

result = pd.DataFrame(predictive_model(data_train,data_test)[0])
result = result.rename(columns={0: 'price_doc'}).sort_index()
result = recti_price(result,0.25)

result.to_csv('~/Dropbox/Data_Science/Kaggle/Sberbank_Russian_Housing_Market/result7.csv',
              index_label='id')
                        

print('#### End computing ####')
