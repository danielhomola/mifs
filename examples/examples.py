"""
Example showing the use of the mifs module.
"""

import mifs
from sklearn.datasets import  make_classification, make_regression
import numpy as np 

def check_selection(selected, i, r):
    """
    Check FN, FP, TP ratios among the selected features.
    """
    # reorder selected features
    try:
        selected = set(selected)
        all_f = set(range(i+r))
        TP = len(selected.intersection(all_f))
        FP = len(selected - all_f)
        FN =  len(all_f - selected)
        if (TP+FN) > 0: 
            sens = TP/float(TP + FN)
        else:
            sens = np.nan
        if (TP+FP) > 0:
            prec =  TP/float(TP + FP)   
        else:
            prec = np.nan
    except:
        sens = np.nan
        prec = np.nan
    return sens, prec
    

if __name__ == '__main__':
    # variables for dataset    
    s = 200
    f = 100
    i = int(.1*f)
    r = int(.05*f)
    c = 2
    
    # simulate dataset with discrete class labels in y
    X, y = make_classification(n_samples=s, n_features=f, n_informative=i, 
                         n_redundant=r, n_clusters_per_class=c, 
                         random_state=0, shuffle=False)
    # perform feature selection
    MIFS = mifs.MutualInformationFeatureSelector(method='JMI', verbose=2)
    MIFS.fit(X,y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(MIFS.support_)[0], i, r)
    print 'Sensitivity: ' + str(sens) + '    Precision: ' + str(prec)
    
    
    # simulate dataset with continuous y 
    X, y = make_regression(n_samples=s, n_features=f, n_informative=i, 
                         random_state=0, shuffle=False)                     
    # perform feature selection
    MIFS = mifs.MutualInformationFeatureSelector(method='JMI', verbose=2, 
                                                  categorical = False)
    MIFS.fit(X,y)
    # calculate precision and sensitivity
    sens, prec = check_selection(np.where(MIFS.support_)[0], i, r)
    print 'Sensitivity: ' + str(sens) + '    Precision: ' + str(prec)