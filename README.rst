.. -*- mode: rst -*-

MIFS
====

Parallelized Mutual Information based Feature Selection module.

Related blog post here_

.. _here: http://danielhomola.com/2016/01/31/mifs-parallelized-mutual-information-based-feature-selection-module/)

Dependencies
------------

* scipy(>=0.17.0)
* numpy(>=1.10.4)
* scikit-learn(>=0.17.1)
* bottleneck(>=1.1.0)

How to use
----------

Download, import and do as you would with any other scikit-learn method:

* ``fit(X, y)``
* ``transform(X)``
* ``fit_transform(X, y)``

Description
-----------

MIFS stands for Mutual Information based Feature Selection. This class contains routines for selecting features using both continuous and discrete y variables. Three selection algorithms are implemented: JMI, JMIM and MRMR.

This implementation tries to mimic the scikit-learn interface, so use fit, transform or fit_transform, to run the feature selection.

See examples/example.py for well examples and usage.

Docs
----

Parameters
~~~~~~~~~~

**method** : string, default = 'JMI'::

  > Which mutual information based feature selection method to use:
  > * 'JMI' : Joint Mutual Information [1]
  > * 'JMIM' : Joint Mutual Information Maximisation [2]
  > * 'MRMR' : Max-Relevance Min-Redundancy [3]

**k** : int, default = 5::

  > Sets the number of samples to use for the kernel density estimation with the kNN method. Kraskov et al. recommend a small integer between 3 and 10.

**n_features** : int or string, default = 'auto'::

  > If int, it sets the number of features that has to be selected from X. If 'auto' this is determined automatically based on the amount of mutual information the previously selected features share with y.

**categorical** : Boolean, default = True::

  > If True, y is assumed to be a categorical class label. If False, y is treated as a continuous. Consequently this parameter determines the method of estimation of the MI between the predictors in X and y.

**verbose** : int, default=0::

  > Controls verbosity of output:
  > * 0: no output
  > * 1: displays selected features
  > * 2: displays selected features and mutual information

Attributes
~~~~~~~~~~

**n_features** : int::

  > The number of selected features.

**support** : array of length [number of features in X]::

  > The mask array of selected features.

**ranking** : array of shape [n_features]::

  > The feature ranking of the selected features, with the first being the first feature selected with largest marginal MI with y, followed by the others with decreasing MI.

**mi** : array of shape n_features::

  > The JMIM of the selected features. Usually this a monotone decreasing array of numbers converging to 0. One can use this to estimate the number of features to select. In fact this is what n_features='auto' tries to do heuristically.

Examples
~~~~~~~~

The following example illustrates the use of the package::

    import pandas as pd
    import mifs

    # load X and y
    X = pd.read_csv('my_X_table.csv', index_col=0).values
    y = pd.read_csv('my_y_vector.csv', index_col=0).values

    # define MI_FS feature selection method
    feat_selector = mifs.MutualInformationFeatureSelector()

    # find all relevant features
    feat_selector.fit(X, y)

    # check selected features
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

References
~~~~~~~~~~

.. [1] H. Yang and J. Moody, "Data Visualization and Feature Selection: New
       Algorithms for Nongaussian Data"
       NIPS 1999
.. [2] Bennasar M., Hicks Y., Setchi R., "Feature selection using Joint Mutual
       Information Maximisation"
       Expert Systems with Applications, Vol. 42, Issue 22, Dec 2015
.. [3] H. Peng, Fulmi Long, C. Ding, "Feature selection based on mutual
       information criteria of max-dependency, max-relevance,
       and min-redundancy"
       Pattern Analysis & Machine Intelligence 2005
