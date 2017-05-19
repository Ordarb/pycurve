__author__= 'Sandro Braun'

import sys
import pandas as pd
import numpy as np
from scipy.optimize import fmin, fmin_l_bfgs_b
from bond import Bond


class Optimization(object):

    def __init__(self, instruments):
        self.instruments = instruments


    def parametric(self, algorithm = 'sv_adj', target='ytm', p_init='grid', w_method='duration',
                 w_short=False):
        '''
        Parameter estimation using parametric models.
        ------------------
        INPUT    algorithm:     'ns' (Nelson-Siegel), 'sv' (Svensson), 'bc' (Boerk-Christensen)
                                or 'sv_adj' (Svensson controlling for heteroscedasticty)
                 target:        'ytm' or 'price' - target in minimization problem
                 p_init:        inital parameter - eiter list of parameter or 'grid search'.
                 w_method:      'duration' or 'uniform' - weighting of individual bonds
                 w_short:       True/False - should have first bond/rate a higher weight  
        ------------------
        RETURNS  params
        '''

        # param initialization
        # optimization algorithm


        # 1) Select the method
        self.algorithm = algorithm
        if algorithm == 'ns':
            self.algo = getattr(self,'NelsonSiegel')
        if algorithm == 'sv':
            self.algo = getattr(self,'Svensson')
        if algorithm == 'sv_adj':
            self.algo = getattr(self,'SvenssonAdj')
        if algorithm == 'bc':
            self.algo = getattr(self,'BoerkChristensen') 

        # 2) Parameter initialization
        params  = self.ParamInitialization(method=p_init)

        # 3) prepare for optimization
        self.w_method   = w_method
        self.w_short    = w_short
                  
        # 4) optimization
        
        if target == 'ytm':
            return self.optimizeSwiss(self.function_ytm, params)
            
        if target == 'price':
            return self.optimizeSwiss(self.function_price, params)

        # differential evolution algorithm
        
        return params
        

    def optimizeSwiss(self, function, params):
        '''
        Optimization method as used from SwissNationalBank (SNB). Two step optimization using
        sequentially the 1) Simplex and 2) Berndt,Hall, Hall and Hausmann (BHHH) algorithms
        to find optimal parameters.
        ------------------
        INPUT    function:  target function to minimize (ytm or price)
                 params:    initial parameter set to start with
        ------------------
        RETURNS  paramss
        '''


        # 1) weighting method
        weights = self.weighting(w_method=self.w_method,w_short=self.w_short)

        # 2) check for starting values of the optimization
        
        if self.multistart == True:
            min_error   = list()
            opt_params  = list()

            for i in range(params.shape[0]):

                p = params[i]
                try:
                    p,error,_,_,warnflag = fmin(function, p, args=(weights,),maxiter=300,full_output=True, disp=False)
                except: pass

                bounds = self.boundaries(p)
                p,error,warnflag = fmin_l_bfgs_b(function, p, args=(weights,),bounds = list(bounds),approx_grad=True)
                opt_params.append(p)
                min_error.append(error)

            parameter = opt_params[np.argmin(min_error)]
            self.parameter = parameter
            return parameter
                
        elif self.multistart == False:

            p = params
            try:
                p,min_error,_,_,warnflag = fmin(function, p, args=(weights,),maxiter=300,full_output=True, disp=False)
            except: pass
            
            bounds = self.boundaries(p)
            p,min_error,warnflag = fmin_l_bfgs_b(function, p, args=(weights,),
                                                  bounds = list(bounds),approx_grad=True)
            self.parameter = p
            return p
        

    #################################################################################################
    #####################################    Zielfunktionen   #######################################
    #################################################################################################

    def DiscountRates(self, zeros, ttm):
        #aus der Spot_rate und der Zeit wird der Discount Factor gerechnet
        return np.exp(-(zeros*ttm))

    def boundaries(self,params):
        '''
        Defines the boundaries for the parameters as they are used by the respective method in
        Nelson-Siegel, Svensson, or adjusted Svensson. 
        ------------------
        INPUT    bonds:         df of all bond characteristics
                 method:        'uniform' or (inverse) 'duration'
                 weight_short:  boolean, give the very first short rate more weight 
        ------------------
        RETURNS  weights
        '''
        short  = self.instruments.ytm.min()*100
        if self.algorithm  == 'ns':
            return ((0,None),(short-params[0],short-params[0]),(None,None),(0,None))
        if (self.algorithm == 'sv') or (self.algorithm == 'sv_adj') or (self.algorithm == 'bc') :
            return ((0,None),(short-params[0],short-params[0]),(None,None),(None,None),(0,None),(0,None))

    def weighting(self, w_method='duration', w_short= True):
        '''
        Weighting of the errors of individual bonds is crucial to solve several problems.
        Changing weights in the optimization can also address a potential heteroscedasticity
        problem. Implmented are uniform and inverse duration weighting, but theoretically,
        it can also be weighted by bid-ask spread, amount outstanding, closeness to par, etc.
        ------------------
        INPUT    bonds:         df of all bond characteristics
                 method:        'uniform' or (inverse) 'duration'
                 weight_short:  boolean, give the very first short rate more weight 
        ------------------
        RETURNS  weights
        '''
        if      w_method        == 'uniform':    weights = np.ones(len(self.instruments))
        elif    w_method        == 'duration':   weights = np.array(1./self.instruments.duration)
        if      w_short         == True:         weights[0] = 100.
        return  weights
        
    def function_ytm(self, params, weights):
        '''
        =========================================================================
        Objective function for optimization problem
            o optimize Yield Difference with duration weighting
            o short rate gets more weight
        =========================================================================
        '''
        
        ytmObs = np.array(self.instruments.ytm)
        distance  = list()
        ytmFit = list()
        
        for i in range(len(self.instruments)):
            cf          = np.array(self.instruments.cf.iloc[i])
            t           = np.array(self.instruments.timing.iloc[i])
            acc         = np.array(self.instruments.accrued.iloc[i])
            freq        = np.array(self.instruments.cpnFreq.iloc[i])
            coupon      = self.instruments.coupon.iloc[i]
            zero        = self.algo(params, t) /100     # select correct algo
            discount    = self.DiscountRates(zero,t)
            pxFit       = sum(discount*cf)
            ytmFit.append(Bond().ytm_short(pxFit, freq, cf, t, acc, px='dirty'))
            
        distance = weights * (ytmFit - ytmObs) *100
        return np.sum(distance**2)

    def function_price(self, params, weights):
        '''
        =========================================================================
        Objective function for optimization problem
            o optimize Yield Difference with duration weighting
            o short rate gets more weight
        =========================================================================
        '''

        pxObs   = np.array(self.instruments.px_dirty)
        pxFit = list()

        for i in range(len(self.instruments)):
            cf          = np.array(self.instruments.cf.iloc[i])
            t           = np.array(self.instruments.timing.iloc[i])
            acc         = np.array(self.instruments.accrued.iloc[i])
            freq        = np.array(self.instruments.cpnFreq.iloc[i])
            coupon      = self.instruments.coupon.iloc[i]
            zero        = self.algo(params, t) /100     # select correct algo
            discount    = self.DiscountRates(zero,t)
            pxFit.append(sum(discount*cf))
            
        distance = weights * (pxFit-pxObs)
        self.distance = distance
        return np.sum(distance**2)

    #############################################################################

    def ParamInitialization(self, method='grid'):
        '''
        Parameter initalization is very important. Currently, there are 2 methods implemented:
        Either entering a given list of factors (values from last date) or selecting 'grid'
        which spans a grid to multistart the optimization process. 
        ------------------
        INPUT    method:        list or 'grid'
        ------------------
        RETURNS  params (array)
        '''

        if not isinstance(method,str):
            self.multistart = False
            self.parameter = method
            return np.array(method)

        if method == 'grid':
                        
            # 1) initialte parameters
            self.multistart = True
            longYTM     = np.log(1 + self.instruments.ytm.max())*100     # continously compounded max yield
            shortYTM    = np.log(1 + self.instruments.ytm.min())*100   # continously compounded min yield
            longTTM     = self.instruments.ttm.max()
            beta0       = [longYTM]
            beta1       = [shortYTM - beta0]
            maxTau      = min(10, 2.*longTTM)

            # 2) optimal parameter for tau 1 and 2
            def Loading(tau):
                f = lambda x : -1*((1-np.exp(-x*tau))/(x*tau)-np.exp(-x*tau))
                return f
            
            lambda0 = 0.6           
            minTau = fmin(Loading(maxTau),lambda0,maxiter=3e5, xtol=1e-5,ftol=1e-5, disp=False)[0]

            # 3) find optimal multistart parameters
            tau1 = (minTau,100)
            lambd = [0.5, 1.0, 2.0]
            lambd = np.array(filter(lambda x: x > minTau, lambd))
            beta2 = [-5]

            if self.algorithm == 'ns':
                RunsCount = len(beta0)*len(beta1)*len(beta2)*len(lambd)
                params = np.zeros((RunsCount, 4))
                i = 0
                for l1 in np.arange(len(lambd)):
                    for b0 in np.arange(len(beta0)):
                        for b1 in np.arange(len(beta1)):
                            for b2 in np.arange(len(beta2)):
                                params[i,:]=[beta0[b0], beta1[b1], beta2[b2], lambd[l1]]
                                i += 1
            else:
             
                tau2 = (minTau,1000)
                gamma = [0.5, 1, 2]
                gamma = np.array(filter(lambda x: x > minTau, gamma))
                beta3 = [5]
                RunsCount = len(beta0)*len(beta1)*len(beta2)*len(beta3)*len(lambd)*len(gamma)
                params = np.zeros((RunsCount, 6))   
                i = 0
                for l2 in np.arange(len(gamma)):
                    for l1 in np.arange(len(lambd)):
                        for b0 in np.arange(len(beta0)):
                            for b1 in np.arange(len(beta1)):
                                for b2 in np.arange(len(beta2)):
                                    for b3 in np.arange(len(beta3)):
                                        params[i,:]=[beta0[b0], beta1[b1], beta2[b2], beta3[b3],
                                                      lambd[l1], gamma[l2]]
                                        i += 1
            self.parameter = np.array(params)
            return np.array(params)

    #############################################################################   

    def NelsonSiegel(self, params, ttm):
        '''
        Calculates the zero rate using the Nelson-Siegel function.
        ------------------
        INPUT    params:    list of the 3 ns parameters
                 ttm:       time to maturity structure of a bond
        ------------------
        RETURNS  zero
        '''
        assert len(params) == 4, 'NS needs 4 parameter.'
        b0, b1, b2 = params[0], params[1], params[2]
        tau1      = params[3]

        term0 = b0
        term1 = b1 * ((1-np.exp(-ttm/tau1))/(ttm/tau1))
        term2 = b2 * ((1-np.exp(-ttm/tau1))/(ttm/tau1)-np.exp(-ttm/tau1))
        return term0 + term1 + term2

    def Svensson(self, params, ttm):
        '''
        Calculates the zero rate using the Nelson-Siegel-Svensson function.
        ------------------
        INPUT    params:    list of the five nss parameters
                 ttm:       time to maturity structure of a bond
        ------------------
        RETURNS  zero
        '''
        assert len(params) == 6, 'SV needs 6 parameter.'
        b0, b1, b2, b3 = params[0], params[1], params[2], params[3]
        tau1, tau2     = params[4], params[5]

        term0 = b0
        term1 = b1 * ((1-np.exp(-ttm/tau1))/(ttm/tau1))
        term2 = b2 * ((1-np.exp(-ttm/tau1))/(ttm/tau1)-np.exp(-ttm/tau1))
        term3 = b3 * ((1-np.exp(-ttm/tau2))/(ttm/tau2)-np.exp(-ttm/tau2))
        return term0 + term1 + term2 + term3

    def SvenssonAdj(self, params, ttm):
        '''
        Calculates the zero rate using an adjusted Nelson-Siegel-Svensson function
        according to De Pooter (2007). It addresses a potential multicollinearity
        problem in the Svensson approach that arises if tau1 and tau2 have similar
        ------------------
        INPUT    params:    list of the five nss parameters
                 ttm:       time to maturity structure of a bond
        ------------------
        RETURNS  zero
        '''
        assert len(params) == 6, 'SV_Adj needs 6 parameter.'
        b0, b1, b2, b3 = params[0], params[1], params[2], params[3]
        tau1, tau2     = params[4], params[5]

        term0 = b0
        term1 = b1 * ((1-np.exp(-ttm/tau1))/(ttm/tau1))
        term2 = b2 * ((1-np.exp(-ttm/tau1))/(ttm/tau1)-np.exp(-ttm/tau1))
        term3 = b3 * ((1-np.exp(-ttm/tau2))/(ttm/tau2)-np.exp(-2*ttm/tau2))
        return term0 + term1 + term2 + term3

    def BoerkChristensen(self, params, ttm):
        '''
        Similar to the Svensson algorithm, but changes the formula with the
        last curvature parameter
        ------------------
        INPUT    params:    list of the five nss parameters
                 ttm:       time to maturity structure of a bond
        ------------------
        RETURNS  zero
        '''
        assert len(params) == 6, 'BC needs 6 parameter.'
        b0, b1, b2, b3 = params[0], params[1], params[2], params[3]
        tau1, tau2     = params[4], params[5]

        term0 = b0
        term1 = b1 * ((1-np.exp(-ttm/tau1))/(ttm/tau1))
        term2 = b2 * ((1-np.exp(-ttm/tau1))/(ttm/tau1)-np.exp(-ttm/tau1))
        term3 = b3 * ((1-np.exp(-(2*ttm)/tau2))/((2*ttm)/tau2))
        return term0 + term1 + term2 + term3
