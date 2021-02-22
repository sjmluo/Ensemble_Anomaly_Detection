from modules.evaluation import EvaluationFramework
from modules.metrics import metrics

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.utils.utility import standardizer

import pandas as pd
import re
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from modules.deep_learning.VAE import ReconstructionVAE, VAErcp, VAEvampprior
from modules.deep_learning.ensemble import Ensemble

methods = {
    'ensemble': Ensemble(),
    'vae': ReconstructionVAE(),
    'vaercp': VAErcp(),
    'vamprior': VAEvampprior(),
    'iforest': IForest(),
    'knn': KNN(),
    'lof': LOF(),
    'pca': PCA(),
    'ocsvm': OCSVM()
}


def run():
    scoring = list(metrics.keys())
    columns = ["dataset", "method",] + scoring
    df = pd.DataFrame(
            columns=columns)
    df.to_csv("results/results.csv",index=False)

    dataset_folder = "datasets/"
    onlyfiles = [join(dataset_folder, f) for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]

    index = 0
    for file in onlyfiles[:11]:
        mat = loadmat(file)
        X = mat['X']
        y = mat['y'].ravel()

        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,train_size=0.6)

        X_train_norm, X_test_norm = standardizer(X_train, X_test)
        for key,model in methods.items():
            eva = EvaluationFramework(model)

            if isinstance(model, Ensemble):
                eva.supervised_fit(X_train_norm, y_train)
            else:
                eva.fit(X_train_norm)
            y_pred = eva.predict(X_test_norm)
            scores = eva.score(y_test,y_pred)

            results = {}
            name = re.search("\/(.*?)\.mat",file).group(1)
            results["dataset"] = name
            results["method"] = key

            results = {**results,**scores}

            temp = pd.DataFrame(results,index=[index])
            df = pd.concat([df,temp])
            df.to_csv("results/results.csv",index=False)

            index += 1
            for ele in ["pca","tsne"]:
                plt.clf()
                eva.visualise(X_test_norm,y=y_pred,method=ele)
                title = f'{key}-{name}-{ele}'
                plt.title(title)
                plt.savefig(f'results/figures/{title}.png',bbox_inches="tight")
                #plt.show()


run()
