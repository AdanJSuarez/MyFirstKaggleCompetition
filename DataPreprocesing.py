####

import pandas as pd
import numpy as np
from math import log
from pandas.tseries.offsets import *


#####################################
####      Helper functions      #####
#####################################

def clean_nan(data, umbral=0.35):
    ''' umbral in per 1 '''
    dataclean = data
    for i in data_test.columns:
        percent = data_test[i].isnull().sum()/len(data)
        if percent >= umbral:
            dataclean = dataclean.drop(i,1)
    return dataclean

def clean_nan_macro(data, umbral = 0.35):
    ''' umbral in per 1 '''
    dataclean = data
    for i in data.columns:
        percent = data[i].isnull().sum()/len(data)
        if percent >= umbral:
            dataclean = dataclean.drop(i,1)
    return dataclean

def feature_with_nan(data):
    dict_nan = {}
    for i in data.columns:
        percent = data[i].isnull().sum()/len(data)
        if percent > 0:
            dict_nan[i] = percent
    return list(dict_nan.keys())

def feature_type(data,datatype):
    feature_list = []
    for i in data.columns:
        if isinstance(data[i][0], datatype):
            feature_list.append(i)
    return feature_list

def mapping_feature(data, feature):
    '''
    This function leave the NaN intact. Interesting when you what
    to interpolate NaNs
    '''
    features_list = []
    notnulls = data[feature].notnull()
    for i,j in zip(notnulls, range(len(data[feature]))):
        if i:
            features_list.append(data[feature][j])
    feature_set = set(features_list)
    mapping_argument = {x:y for y,x in enumerate(feature_set, start=1)}
    data[feature] = data[feature].map(mapping_argument)#.astype(int)


def complete_nan(data):
    features = feature_with_nan(data)
    n = 0
    for i in features:
        print('# Completing NaNs in',i,len(features)-n)
        datatype = type(data[i][0])
        ind = data[i][data[i].isnull()].index
        mu = data[i].describe()[1]
        sigma = data[i].describe()[2]
        if datatype == np.int64:
            data[i].ix[ind] = np.int64(abs(np.random.normal(mu,sigma,len(ind))))
        elif datatype == np.float64:
            data[i].ix[ind] = np.float64(abs(np.random.normal(mu,sigma,len(ind))))
        elif datatype == int:
            data[i].ix[ind] = int(abs(np.random.normal(mu,sigma,len(ind))))
        elif datatype == float:
            data[i].ix[ind] = float(abs(np.random.normal(mu,sigma,len(ind))))
        n += 1

def feature_type(data,data_type):
    feature_list = []
    for i in data.columns:
        if isinstance(data[i][0], data_type):
            feature_list.append(i)
    return feature_list

def clean_str(data):
    dataclean = data
    for i in data.columns:
        if isinstance(data[i][0], str):
            dataclean = dataclean.drop(i,1)
    return dataclean

def include_features(data,macro):
    datanew = pd.DataFrame(columns=(data.ix[0].append(macro.ix[0])).index)
    n = 0
    for i in macro.index:
        if i in data['timestamp'].values:      
            print(i, ' ', len(macro.index)-n, end="\r")
            val1 = data.loc[data['timestamp'] == i]
            for j in val1.index:
                datanew.loc[j] = data.ix[j].append(macro.ix[i])
            n += 1
    return datanew

###################################
####    END HELPER FUNCTION    ####
###################################

print('##### Start computing #####')

# Loading data
ad_data_train = '/home/ajsg/Documents/Dataset/Kaggle/SberbankRussianHousingMarket/train.csv'
ad_data_test = '/home/ajsg/Documents/Dataset/Kaggle/SberbankRussianHousingMarket/test.csv'
ad_data_macro = '/home/ajsg/Documents/Dataset/Kaggle/SberbankRussianHousingMarket/macro.csv'
data_macro = pd.read_csv(ad_data_macro, index_col='timestamp')
data_train = pd.read_csv(ad_data_train)
data_test = pd.read_csv(ad_data_test)

##### Cleaning data

# Droping features with too many NaN
data_train = clean_nan(data_train)
data_test = clean_nan(data_test)
data_macro = clean_nan_macro(data_macro)

# Droping str features in macro
data_macro = clean_str(data_macro)

# Mapping String features
str_features = feature_type(data_train, str)

for i in str_features:
    print('Do you want to map', i, '?')
    print('Yes--> y / Not--> n')
    s = input('--> ')
    if s == 'y':
        mapping_feature(data_train, i)
        mapping_feature(data_test, i)

# Interpolating missing data (NaN)

complete_nan(data_train)
complete_nan(data_test)
complete_nan(data_macro)
 

# Dropping the bigger one because it is likely a mistake and full_sq = 0
data_train = data_train.drop(3527)
data_train = data_train.drop(data_train.loc[data_train['full_sq'] == 0].index)

#### Featuring engineering
data_train['Price/full_sq'] = data_train['price_doc']/data_train['full_sq']

# Include macro data in train and test
data_train = include_features(data_train, data_macro)
data_test = include_features(data_test, data_macro)

# Mapping timestamp
#mapping_feature(data_train, 'timestamp')
#mapping_feature(data_test, 'timestamp') 

#### Saving cleanded data
data_train.to_csv('/home/ajsg/Documents/Dataset/Kaggle/SberbankRussianHousingMarket/trainclean.csv',index = False)
data_test.to_csv('/home/ajsg/Documents/Dataset/Kaggle/SberbankRussianHousingMarket/testclean.csv',index = False)
data_macro.to_csv('/home/ajsg/Documents/Dataset/Kaggle/SberbankRussianHousingMarket/macroclean.csv')


print('##### End computing #####')
