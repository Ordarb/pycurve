------------------------------
pycurve 0.1.0

Interest Rate Term Structure Modelling
------------------------------

This module allows for interest rate curve estimation by applying the standard approaches in the academic
literature. Within the module there are parametric or spline based models to choose from. To estimate the 
term structure of interest rates, most central banks (9 out of 14) use either the Nelson and Siegel (1987)
or the extended version suggested by Svensson (1994). Exceptions are Canada, Japan, the United Kingdom, 
and the United States which all apply variants of the “smoothing splines” method.


Parametric models:
Are function-based models, using a single-piece function to define the entire maturity domain. After defining
a certain functional form, the model parameters are chosen an optimization. Most significant contributions are
the Nelson-Siegel (1987) model (NS) and several of its enhancements like the Svensson (1994) or Björk-Christensen 
(1999) model. 

Spline-based models:
Fit the yield curve by relying on a piecewise polinomial (spline function), where the individual segments are
joined smoothly at the so-called knot points. Over a closed interval, a given continuous function can be 
approximated by selecting an arbitrary polynomial, where the goodness-of-fit increases with the order of the 
polynomial. Significant contributions are the smoothing-spline method, an extension of the cubic spline technique.
These models are currently under construction and not available in version 0.1.

TODO:

Parametric
- Larger Error summary (RMSEs, etc.)
- Add short rates / better data import
- extract par yield
- differential evolution algorithm for optimization -> global minimum

Spline:
- make it work correctly
- speed it up
- implementation of other penalty terms (FNZ, etc.)
- formula for penalty term if optim. target is 'ytm'?
