This repository contains mandatory assigments written together with Sebastian Hørlück for the course AØKA08084U Advanced Microeconometrics at University of Copenhagen taught by Jesper Riis-Vestergaard Sørensen and Anders Munk-Nielsen, see: https://kurser.ku.dk/course/a%c3%98ka08084u/2021-2022. It also contains the Python implementation of the estimators that we are taught. Most of the code is taken or reused from exercise but it has been has been re-designed to be object-oriented with an experience similar to packages like `Statsmodels` or `ScikitLearn` with a `.fit()` and `.predict()` method for example. Other functionality that has been added is the formula approach to model specification known from `R`, i.e. `y ~ x1 + x2` with the possibility to specify interaction terms and polynomial features, more informative output tables and the statistical tests have been altered to work more generally. It should also be mentioned that the course and also the implementation is to a large extent informed by the fantastic books: *Econometric Analysis of Cross Section and Panel Data (2010)* by Jeffrey M. Wooldridge and *The Elements of Statistical Learning (2009)* by  Trevor Hastie, Robert Tibshirani and Jerome Friedman that are used throughout the course. Some of the included estimators are:

1. `Linear panel models`
    1. `Pooled OLS`
    2. `Fixed Effects`
    3. `First Difference`
    4. `Random Effects`
2. `High dimensional models`
3. `Non-linear models`

And some statistical tests:

1. `Test for serial correlation in panel models`
2. `The Hausman test`
3. `Test for strict exogenity in panel models`
4. `The wald test`
