# train the SVR model

import pickle
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class MLmodel(object):

    def __init__(self, tdatapath, outpath):

        self.tdatapath = tdatapath
        self.trainingdata = pickle.load(open(self.tdatapath, 'rb'))
        self.outpath = outpath

    def trainSVR(self):

        # soil moisture
        target = self.trainingdata[:,3]
        valid = np.where(target < 80)
        target = target[valid]
        self.target = target
        # features
        features = np.copy(self.trainingdata[valid,6::]).squeeze()
        self.features = features

        # valid features
        valid = np.where(np.any(features != -9999, axis=1))
        self.target = self.target[valid]
        self.features = self.features[valid,]

        # scaling
        scaler = StandardScaler().fit(features)
        features = scaler.transform(features)
        self.scaler = scaler

        # splitting data into training and test-set
        x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                            test_size=0.1,
                                                            train_size=0.9)

        x_train = features
        y_train = self.target
        x_test = features
        y_test = self.target

        # define parameter grid for grid search
        dictCV = dict(C=np.logspace(-2, 2, 10),
                      gamma=np.logspace(-2, -0.5, 10),
                      epsilon=np.logspace(-2, -0.5,10),
                      kernel=['rbf'])

        # specify learning routine
        svr_rbf = SVR()

        # specify kernel and serach settings settings
        gdCV = GridSearchCV(estimator=svr_rbf,
                             param_grid=dictCV,
                             n_jobs=8,
                             verbose=1,
                             pre_dispatch='all',
                             cv=KFold(n_splits=5, shuffle=True, random_state=42),
                             #cv=LeaveOneOut(),
                             scoring='r2')

        # Fit
        gdCV.fit(x_train, y_train)
        self.SVRmodel = gdCV
        y_CV_rbf = gdCV.predict(x_test)

        r = np.corrcoef(y_test, y_CV_rbf)
        error = np.sqrt(np.sum(np.square(y_test - y_CV_rbf)) / len(y_test))

        #print(r)
        #print(error)

       # pickle.dump((gdCV, scaler), open(self.outpath + 'mlmodel.p', 'wb'))

        print('SVR performance based on test-set')
        print('R: ' + str(r[0, 1]))
        print('RMSE. ' + str(error))

        pltlims = 70

        # create plots
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_CV_rbf, c='g', label='True vs Est')
        plt.xlim(0, pltlims)
        plt.ylim(0, pltlims)
        plt.xlabel("In-Situ SMC [m3m-3]")
        plt.ylabel("Estimated SMC [m3m-3]")
        plt.plot([0, pltlims], [0, pltlims], 'k--')
        plt.savefig(self.outpath + 'truevsest_pythonSVR_S1.png')
        plt.close()


def execute_training(trainingdata, outpath):

    ModelDict = MLmodel(trainingdata, outpath)
    ModelDict.trainSVR()

    pickle.dump(ModelDict, open(outpath + 'SVR_Model_Python_S1.p', 'wb'))