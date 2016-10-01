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

* fit(X, y)
* transform(X)
* fit_transform(X, y)

Descriptio
----------

MIFS stands for Mutual Information based Feature Selection. This class contains routines for selecting features using both continuous and discrete y variables. Three selection algorithms are implemented: JMI, JMIM and MRMR.

This implementation tries to mimic the scikit-learn interface, so use fit, transform or fit_transform, to run the feature selection.

See examples/example.py for well examples and usage.
