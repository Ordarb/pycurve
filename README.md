------------------------------
pycurve 0.1.0

Yield Curve Estimation
------------------------------

This module allows for yield curve estimation by applying the standard approaches in the academic literature. 
Currently implemented are parametric as well as spline based approaches. To estimate the term structure of 
interest rates, most central banks (9 out of 14) use either Nelson and Siegel (1987) or the extended version 
as suggested by Svensson (1994). Exceptions are Canada, Japan, the United Kingdom, and the United States which 
all apply variants of the “smoothing splines” method, including penalties for non-smoothness of the instantanous 
forward rate.

Parametric models:
Are function-based models, using single-piece functions to define the entire maturity domain. After defining
a certain functional form, the model parameters are chosen an optimization. The most significant contributions are
the Nelson-Siegel model (NS) and several of its enhancements like the Svensson (SV) or Björk-Christensen (BC) model. 

Spline-based models:
Fit the yield curve by relying on a piecewise polinomial (spline function), where the individual segments are
joined smoothly at the so-called knot points ("smoothing splines"). Over a closed interval, a given continuous 
function can be approximated by selecting an arbitrary polynomial, where the goodness-of-fit increases with the order 
of the polynomial. Significant contributions in the yield curve literature is the smoothing-spline method, an extension 
of the cubic spline technique. Hereby the smoothing splines include a penalty for non-smooth behaviour of the 
instantanous forward rate (roughness penalty), like in the "Variable Roughness Penalty" approach, used by the
Bank of England.

--------------------------------------------------------------------

Example:

Load data
.. code-block:: pycon
import pycurve
>>> data1 = pycurve.Data().db_data('SWISS','CHF','newest')
>>> data2 = pycurve.Data().bb_srch('SRCH_SWISS)

Estimate yield curve
.. code-block:: pycon
>>> c1 = pycurve.Curve().Parameteric(algorithm='ns',target='ytm', w_method='duration')    # nelson-siegel
>>> c2 = pycurve.Curve().Parameteric(algorithm='sv',target='ytm', w_method='duration')    # svensson
>>> c3 = pycurve.Curve().Parameteric(algorithm='bc',target='ytm', w_method='duration')    # björn-christensen
>>> c4 = pycurve.Curve().SmoothingSpline(algorithm='VRP', target='price', w_method='duration', knots=[7,15]

Show results
.. code-block:: pycon
>>> c1.plotting([c1,c2,c3,c4])

----------------------------------------------------------------------








TODO (version 0.2.0):

General
- Add short rates / better data import
- extract par yield necessary?
- Larger Error summary (RMSEs, etc.)
- faster minimization algorithm than simplex

Parametric
- differential evolution algorithm for optimization -> global minimum
- SvenssonAdj fwd calculation is wrong

Spline:
- speed it up
- implementation of other penalty terms (FNZ, etc.)
- formula for penalty term if optim. target is 'ytm'?
