import numpy as np
from scipy.io.arff import loadarff
import pandas as pd
from numpy import rint

def data(row = False):
    return data_preprocessing(rawdf())

def rawdf():
    arff = loadarff('src/data/seismic-bumps.arff')
    train =  pd.DataFrame(arff[0])
    stringcols = ['seismic','seismoacoustic', 'shift', 'ghazard', 'class']
    for col in stringcols:
        train[col] = train[col].apply(lambda x: x.decode('utf8'))

    def encodeshift(s):
        if s == 'W':
            return 0
        elif s == 'N':
            return 1

    train['shift'] = train['shift'].apply(encodeshift)
    
    d = {'a':0, 'b':1, 'c':2, 'd':3}
    def encodeseismic(s):
        return d[s]

    train['gdenergy'] = ((train['gdenergy'] - train['gdenergy'].mean())/train['gdenergy'].std())
    train['gdpuls'] = ((train['gdpuls'] - train['gdpuls'].mean())/train['gdpuls'].std())

    train['seismic'] = train['seismic'].apply(encodeseismic)
    train['seismoacoustic'] = train['seismoacoustic'].apply(encodeseismic)
    train['ghazard'] = train['ghazard'].apply(encodeseismic)
    
    return train        

def data_preprocessing(train):

    def encodeshift(s):
        if rint(s).astype('int32') == 0:
            return np.array([1,0])
        elif rint(s).astype('int32') == 1:
            return np.array([0,1])

    train.loc[:,'shift'] = train['shift'].apply(encodeshift)
    
    d = {'a':0, 'b':1, 'c':2, 'd':3}
    def encodeseismic(s):
        res = np.zeros(4)
        res[rint(s).astype('int32')] = 1
        return res


    train.loc[:,'seismic'] = train['seismic'].apply(encodeseismic)
    train.loc[:,'seismoacoustic'] = train['seismoacoustic'].apply(encodeseismic)
    train.loc[:,'ghazard'] = train['ghazard'].apply(encodeseismic)

    train.loc[:,'genergy'] = np.log(train['genergy'])
    train.loc[:,'gpuls'] = np.log(train['gpuls'])
    train.loc[:,'energy'] = np.log(train['energy'] + 1)
    train.loc[:,'maxenergy'] = np.log(train['maxenergy'] + 1)

    train['nbumps49'] = train.loc[:,['nbumps4','nbumps5','nbumps6','nbumps7','nbumps89']].sum(1)

    def vectorisenb(row):
        return np.array(row.loc[['nbumps','nbumps2','nbumps3','nbumps49']]).astype('float32')
    train['nbumpsv'] = train.apply(vectorisenb,1)

    train['class'] = train['class'].astype(int)

    train = train.loc[:,['gdenergy', 'gdpuls', 'genergy', 'energy', 'maxenergy', 'gpuls', 'shift', 'seismic', 'seismoacoustic', 'ghazard', 'nbumpsv', 'class']]

    #if row: return np.array(train)[:,:-1], np.array(train)

    trainlist = [None for x in range(len(train.columns))]

    for x in [0,1,2,3,4,5,11]:
        trainlist[x] = np.expand_dims(np.asarray(train.iloc[:,x]).astype('float32'), -1)
    
    for x in range(6,11):
        trainlist[x] = np.asarray(np.stack(train.iloc[:,x])).astype('float32')

    trainlist[-1] = trainlist[-1].astype('int32')

    return trainlist[:-1], trainlist


if __name__ == "__main__":
    print(data()[1])