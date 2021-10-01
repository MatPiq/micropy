This repository contains mandatory written assigments in AØKA08084U Advanced Microeconometrics at University of Copenhagen and the Python implementation of the estimators that have been taught. See: https://kurser.ku.dk/course/a%c3%98ka08084u/2021-2022. Most of the code is taken or reused from exercise but it has been has been re-designed to be object-oriented with an experience similar to packages like `Statsmodels` or `ScikitLearn` with a `.fit()` and `.predict()` method for example. Other functionality that has been added is the formula approach to model specification known from `R`, i.e. `y ~ x1 + x2` with the possibility to specify interaction terms and polynomial features. Included estimators are:

1. `Linear panel models`
    1. `Pooled OLS`
    2. `Fixed Effects`
    3. `First Difference`
    4. `Random Effects`
2. `High dimensional models`
3. `Non-linear models`

