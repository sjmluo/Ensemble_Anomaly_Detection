import numpy as np
from scipy.io.arff import loadarff
import pandas as pd
def data():
    arff = loadarff('src/data/seismic-bumps.arff')
    train = pd.DataFrame(arff[0])

    train['gdenergy'] = ((train['gdenergy'] - train['gdenergy'].mean())/train['gdenergy'].std())
    train['gdpuls'] = ((train['gdpuls'] - train['gdpuls'].mean())/train['gdpuls'].std())

    stringcols = ['seismic','seismoacoustic', 'shift', 'ghazard', 'class']
    for col in stringcols:
        train[col] = train[col].apply(lambda x: x.decode('utf8'))

    def encodeshift(s):
        if s == 'W':
            return np.array([1,0])
        elif s == 'N':
            return np.array([0,1])

    train['shift'] = train['shift'].apply(encodeshift)
    
    d = {'a':0, 'b':1, 'c':2, 'd':3}
    def encodeseismic(s):
        res = np.zeros(4)
        res[d[s]] = 1
        return res

    train['seismic'] = train['seismic'].apply(encodeseismic)
    train['seismoacoustic'] = train['seismoacoustic'].apply(encodeseismic)
    train['ghazard'] = train['ghazard'].apply(encodeseismic)

    train['genergy'] = np.log(train['genergy'])
    train['gpuls'] = np.log(train['gpuls'])
    train['energy'] = np.log(train['energy'] + 1)
    train['maxenergy'] = np.log(train['maxenergy'] + 1)

    train['nbumps49'] = (train.loc[:,['nbumps4','nbumps5','nbumps6','nbumps7','nbumps89']].sum(1))

    def vectorisenb(row):
        return np.array(row.loc[['nbumps','nbumps2','nbumps3','nbumps49']])
    train['nbumpsv'] = train.apply(vectorisenb,1)

    train['class'] = train['class'].astype(int)

    train = train.loc[:,['gdenergy', 'gdpuls', 'genergy', 'energy', 'maxenergy', 'gpuls', 'shift', 'seismic', 'seismoacoustic', 'ghazard', 'nbumpsv', 'class']]

    trainlist = [None for x in range(len(train.columns))]

    for x in [0,1,2,3,4,5,11]:
        trainlist[x] = np.expand_dims(np.asarray(train.iloc[:,x]).astype('float32'), -1)
    
    for x in range(6,11):
        trainlist[x] = np.asarray(np.stack(train.iloc[:,x])).astype('float32')

    return trainlist[:-1], trainlist

if __name__ == "__main__":
    print(data()[1])