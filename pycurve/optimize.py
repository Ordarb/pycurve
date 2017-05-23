__author__= 'Sandro Braun'


import sys
import pandas as pd
import numpy as np
from scipy.optimize import fmin, fmin_l_bfgs_b
from bond import Bond


params = np.array([0.55, -1.35, -13.6, 11.42, 1.15, 0.94])


class Optimization(object):

    def __init__(self, instruments):
        self.instruments = instruments



    def SmoothSpline(self):
        pass



    def Parametric(self, algorithm = 'sv_adj', target='ytm', p_init='grid', w_method='duration',
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
        RETURNS  None
        '''

        # 1) Select the method
        self.algorithm  = algorithm
        self._target     = target
        self._p_init     = p_init
        if algorithm == 'ns':
            self._algo = getattr(self,'_NelsonSiegel')
        if algorithm == 'sv':
            self._algo = getattr(self,'_Svensson')
        if algorithm == 'sv_adj':
            self._algo = getattr(self,'_SvenssonAdj')
        if algorithm == 'bc':
            self._algo = getattr(self,'_BoerkChristensen') 

        # 2) Parameter initialization
        params  = self._ParamInitialization(method=p_init)

        # 3) prepare for optimization
        self._w_method   = w_method
        self._w_short    = w_short
                  
        # 4) optimization
        if target == 'ytm':
            self._optimizeSwiss(self._function_ytm, params)
        
        if target == 'price':
            self._optimizeSwiss(self._function_price, params)

        self._summary()
        
        

    def _optimizeSwiss(self, function, params):
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

        #TODO: abfangen falls es nicht konvergiert!!!

        # 1) weighting method
        weights = self._weighting(w_method=self._w_method,w_short=self._w_short)

        # 2) check for starting values of the optimization
        
        if self._multistart == True:
            min_error   = list()
            opt_params  = list()

            for i in range(params.shape[0]):

                p = params[i]
                try:
                    p,error,_,_,warnflag = fmin(function, p, args=(weights,),maxiter=300,full_output=True, disp=False)
                except: pass

                bounds = self._boundaries(p)
                p,error,warnflag = fmin_l_bfgs_b(function, p, args=(weights,),bounds = list(bounds),approx_grad=True)
                opt_params.append(p)
                min_error.append(error)

            parameter = opt_params[np.argmin(min_error)]
            self.parameter = parameter
            function(parameter,weights)         # for correct optimization errors
            self._createZero(ttm=None, t_max=30)
            
                
        elif self._multistart == False:

            p = params
            try:
                p,min_error,_,_,warnflag = fmin(function, p, args=(weights,),maxiter=300,full_output=True, disp=False)
            except: pass
            
            bounds = self._boundaries(p)
            p,min_error,warnflag = fmin_l_bfgs_b(function, p, args=(weights,),
                                                  bounds = list(bounds),approx_grad=True)
            self.parameter = p
            self._createZero(ttm=None, t_max=30)
            


    def _summary(self):
        '''
        Samples results of the model for better use of output and details. Provides informations
        about the model, the output and the the error (fit of the model with the underlying bonds).
        ------------------
        '''
        self.model  = {'model':self.algorithm, 
                       'optimization': {'objectiv function': self._target, 'param_init':self._p_init,
                                        'weights': self._w_method, 'ShortRate_weighting':self._w_short}}
        self.date   = self.instruments.date.iloc[0]
        self.output = {'parameter': self.parameter}
        self.error  = {'RMSE': self._RMSE, 'MAE': self._MAE}
        self.zero   = None
        self.fwd    = None
        self.par    = None
        

    #################################################################################################
    #####################################    Zielfunktionen   #######################################
    #################################################################################################




    def _DiscountRates(self, zeros, ttm):
        #aus der Spot_rate und der Zeit wird der Discount Factor gerechnet
        return np.exp(-(zeros*ttm))

    def _boundaries(self,params):
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

        

    def _weighting(self, w_method='duration', w_short= True):
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
        

    def _function_ytm(self, params, weights):
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
            zero        = self._algo(params, t) /100     # select correct algo
            discount    = self._DiscountRates(zero,t)
            pxFit       = sum(discount*cf)
            ytmFit.append(Bond().ytm_short(pxFit, freq, cf, t, acc, px='dirty'))
            
        distance = weights * (ytmFit - ytmObs) *100

        # error measures
        self._RMSE = np.sqrt(np.mean((ytmFit-ytmObs)**2))
        self._MAE  = np.mean(np.abs(ytmFit-ytmObs))

        return np.sum(distance**2)


    def _function_price(self, params, weights):
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
            zero        = self._algo(params, t) /100     # select correct algo
            discount    = self._DiscountRates(zero,t)
            pxFit.append(sum(discount*cf))
            
        distance = weights * (pxFit-pxObs)
        self.distance = distance

        # error measures
        self._RMSE = np.sqrt(np.mean((pxFit-pxObs)**2))
        self._MAE  = np.mean(np.abs(pxFit-pxObs))

        return np.sum(distance**2)

    #############################################################################



    def _ParamInitialization(self, method='grid'):
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
            self._multistart = False
            self.parameter = method
            return np.array(method)

        if method == 'grid':
                        
            # 1) initialte parameters
            self._multistart = True
            longYTM     = np.log(1 + self.instruments.ytm.max())     # continously compounded max yield
            shortYTM    = np.log(1 + self.instruments.ytm.min())   # continously compounded min yield
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






   


    def _createZero(self, ttm=None, t_max=30):
        '''
        Creates Zero Yields from the parameters estimated. Can chose zero yields
        of 'bonds' or of a DateRange up to 30:
        ------------------
        INPUT    ttm:     None or 'bonds'
                 t_max:   int
        ------------------
        RETURNS  zero
        '''


        if isinstance(ttm,list):
            self.zero = self._algo(self.parameter,np.arange(1,t_max,1))
        if ttm == None:
            if self.instruments.ttm.max() > t_max:
                t_max = self.instruments.ttm.max()
            self.zero = self._algo(self.parameter,np.arange(1,int(t_max)+1,1))

    def _createFwd(self):
        pass

    def _createPar(self):
        pass

                    



    #############################################################################   

    def _NelsonSiegel(self, params, ttm):
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

    def _Svensson(self, params, ttm):
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

    def _SvenssonAdj(self, params, ttm):
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

    def _BoerkChristensen(self, params, ttm):
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
        
    #############################################################################