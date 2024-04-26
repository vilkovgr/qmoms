'''
MIT License

Copyright (c) 2008-2024 Grigory Vilkov

Contact: vilkov@vilkov.net www.vilkov.net

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# import packages and variables
import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
import warnings

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# *****************************************************************************
# compute the moments from a dataframe-based group
# *****************************************************************************
def qmoms_compute_bygroup(groupparams,
                          id=None, rate=None, days=None, date=None,
                          cols_map={'id': 'id',
                                    'date': 'date',
                                    'days': 'days',
                                    'rate': 'rate',
                                    'mnes': 'mnes',
                                    'impl_volatility': 'impl_volatility'}):
    """
    Computes implied moments for grouped option data using specified parameters and column mappings.

    Parameters:
    - groupparams (tuple, list, or dict): If a tuple or list, this should contain the group data and parameters as the
    first and second elements, respectively. If a dict, it is used directly as parameters.
    - id (int, optional): Identifier for the data group, defaults to the first 'id' in the group data as specified in
    cols_map if not provided.
    - rate (float, optional): Risk-free interest rate, defaults to the first 'rate' in the group data as specified in
    cols_map if not provided.
    - days (int, optional): Days until expiration, defaults to the first 'days' in the group data as specified in cols_
    map if not provided.
    - date (any, optional): Date of the data, defaults to the first 'date' in the group data as specified in cols_map
    if not provided.
    - cols_map (dict, optional): A dictionary mapping the expected columns in the group data to the actual column names.
    Default is {'id': 'id', 'date': 'date', 'days': 'days', 'rate': 'rate',
                'mnes': 'mnes', 'impl_volatility': 'impl_volatility'}.

    Returns:
    - pandas.Series: Contains the computed moments along with initial group identifiers such as id, date, and days.

    The function sorts the group data by moneyness, removes duplicates, and computes moments using interpolation.
    It then merges the computed moments with initial data identifiers and returns a pandas Series, using the column
    mappings specified in cols_map for accessing data fields.
    """
    cols_map = cols_map | {'id': 'id',
                           'date': 'date',
                           'days': 'days',
                           'rate': 'rate',
                           'mnes': 'mnes',
                           'impl_volatility': 'impl_volatility'}

    if (isinstance(groupparams, tuple) or isinstance(groupparams, list)) and len(groupparams) == 2:
        group = groupparams[0]
        params = groupparams[1]
    else:
        params = groupparams

    # scalars
    id = id or group[cols_map['id']].iloc[0]
    date = date or group[cols_map['date']].iloc[0]
    days = days or group[cols_map['days']].iloc[0]
    rate = rate or group[cols_map['rate']].iloc[0]

    # the surface is given in two columns
    group = group.sort_values(by=[cols_map['mnes']])
    mnes = group[cols_map['mnes']]
    vol = group[cols_map['impl_volatility']]

    # remove duplicated moneyness points
    goods = ~mnes.duplicated()
    mnes = mnes[goods]
    vol = vol[goods]

    # pre-define the output dict
    res = {'id': id, 'date': date, 'days': days}

    # compute the moments with interpolation
    res_computed = qmoms_compute(mnes, vol, days, rate, params, output='dict')

    # update the output
    res = res | res_computed

    return pd.Series(res)


# *****************************************************************************
# % compute the moments for one id/date/days combo
# *****************************************************************************
def qmoms_compute(mnes, vol, days, rate, params, output='pandas'):
    """
    Computes implied moments for option pricing using interpolated volatility and various moment formulas.

    Parameters:
    - mnes (array-like): moneyness (K/S) of the options.
    - vol (array-like): implied volatilities corresponding to the moneyness.
    - days (int): Days until the options' expiration.
    - rate (float): Risk-free interest rate.
    - params (dict): Configuration parameters containing grid settings and other options.
    - output (str): Specifies the output format ('pandas' for a pandas.Series, else returns a dictionary).

    Returns:
    - pandas.Series or dict: Calculated moments and optionally other metrics based on the provided parameters.

    The function performs the following:
    - Sorts inputs by moneyness.
    - Computes forward prices and interpolation of implied volatilities.
    - Uses Black or Black-Scholes formulas to compute option prices.
    - Calculates implied variances and semi-variances using specified methods.
    - Optionally computes advanced metrics like MFIS/MFIK, Corridor VIX, RIX, Tail Loss Measure,
    and Slopes based on user configurations in `params`.

    #######################################################################################################
    default_params = {'atmfwd': False,  # if True, use FWD as ATM level and Black model for option valuation
                  'grid': {'number_points': 500,  # points*2 + 1
                           'grid_limit': 2},  # 1/(1+k) to 1+k; typically, k = 2
                  'filter': {'mnes_lim': [0, 1000],  # [0.7, 1.3],
                             'delta_put_lim': [-0.5 + 1e-3, 0],  # typically, only OTM
                             'delta_call_lim': [0, 0.5],  # typically, only OTM
                             'best_bid_zero': -1,  # if negative, no restriction on zero bid;
                             'open_int_zero': -1,  # if negative, no restriction on zero open interest;
                             'min_price': 0},  # a limit on the min mid-price
                  'semivars': {'compute': True},  # save semivariances or not, they are computed anyway
                  'mfismfik': {'compute': True},  # compute skewness/ kurtosis
                  'cvix': {'compute': True,  # compute corridor VIX
                           'abs_dev': [0.2],  # absolute deviation from ATM level of 1
                           'vol_dev': [2]},  # deviation in terms of own sigmas from ATM level of 1
                  'rix': {'compute': True},  # compute RIX (Gao,Gao,Song RFS) or not,
                  'tlm': {'compute': True,  # compute Tail Loss Measure (Xiao/ Vilkov) or not
                          'delta_lim': [20],  # OTM put delta for the threshold (pos->neg or neg)
                          'vol_lim': [2]},  # deviation to the left in terms of sigmas from ATM level of 1
                  'slope': {'compute': True,
                            'deltaP_lim': [-0.5, -0.05],
                            # limits to compute the slopedn as Slope from  IV = const + Slope * Delta
                            'deltaC_lim': [0.05, 0.5]
                            # limits to compute the slopeup as -Slope from IV = const + Slope * Delta
                            }
                  }
    #######################################################################################################
    To modify parameters, update the default_params dictionary with the dictionary with desired parameters
    for example, in case you do not want to compute the tlm, and want to change slope parameters, use
    new_params0 = {'tlm': {'compute': False}}
    new_params1 = {'slope': {'compute': True,
                                'deltaP_lim': [-0.4, -0.1],
                                # limits to compute the slopedn as Slope from  IV = const + Slope * Delta
                                'deltaC_lim': [0.05, 0.5]
                                # limits to compute the slopeup as -Slope from IV = const + Slope * Delta
                                }}
    params = default_params | new_params0 | new_params1
    #######################################################################################################
    """
    mnes = np.array(mnes)
    vol = np.array(vol)
    ii = np.argsort(mnes)
    mnes = mnes[ii]
    vol = vol[ii]

    # assign grid parameters to variables
    gridparams = params.get('grid', {})
    gridparams_m = gridparams.get('number_points', 500)
    gridparams_k = gridparams.get('grid_limit', 2)
    grid = np.array([gridparams_m, gridparams_k])
    ##########################################
    # define maturity and forward price (for S0=1)
    ##########################################
    mat = days / 365
    er = np.exp(mat * rate)
    ##########################################
    # compute the moments WITH INTERPOLATION
    ##########################################
    nopt = len(mnes)
    dicct = {'nopt': nopt}

    if (nopt >= 4) & (len(np.unique(mnes)) == len(mnes)):
        #######################################################################
        # interpolate
        iv, ki = interpolate_iv_by_moneyness(mnes, vol, grid)
        # calculate BS prices - OTM calls - mnes >=1
        otmcalls = ki >= 1
        otmputs = ~otmcalls
        kicalls = ki[otmcalls]

        if params.get('atmfwd', False):  # use Black formula
            # OTM calls
            currcalls, itmputs = Black(1, kicalls, rate, iv[otmcalls], mat)
            # OTM puts
            kiputs = ki[otmputs]
            itmcalls, currputs = Black(1, kiputs, rate, iv[otmputs], mat)
        else:
            # OTM calls
            currcalls, itmputs = BlackScholes(1, kicalls, rate, iv[otmcalls], mat)
            # OTM puts
            kiputs = ki[otmputs]
            itmcalls, currputs = BlackScholes(1, kiputs, rate, iv[otmputs], mat)

        #######################################################################
        # use the gridparams to define some variables
        m = gridparams_m  # use points -500:500, i.e. 1001 points for integral approximation
        k = gridparams_k  # use moneyness 1/(1+2) to 1+2 - should be enough as claimed by JT
        u = np.power((1 + k), (1 / m))

        mi = np.arange(-m, m + 1)  # exponent for the grid
        mi.shape = otmcalls.shape
        ic = mi[otmcalls]  # exponent for OTM calls
        ip = mi[otmputs]

        ### compute some general input to implied variances
        Q = np.append(currputs, currcalls)  # all option prices
        Q.shape = (len(Q), 1)
        dKi = np.empty((len(Q), 1))  # dK
        dKi[:] = np.NAN
        x = (ki[2:, ] - ki[0:-2]) / 2
        x = np.reshape(x, [ki.shape[0] - 2, 1])
        dKi[1:-1] = x
        dKi[0] = ki[1] - ki[0]
        dKi[-1] = ki[-1] - ki[-2]
        dKi = abs(dKi)

        #######################################################################
        ### compute SMFIV -- Ian Martin
        K0sq = 1
        # inputs for semivars:
        svar_ingredients = np.divide(np.multiply(dKi, Q), K0sq)
        svar_multiplier = 2 * er
        # semivariance
        moments_smfivu = svar_multiplier * np.sum(svar_ingredients[otmcalls]) / mat
        moments_smfivd = svar_multiplier * np.sum(svar_ingredients[otmputs]) / mat
        # total variance
        moments_smfiv = moments_smfivu + moments_smfivd

        #######################################################################
        ### compute MFIV_BJN - Britten-Jones and Neuberger (2002)
        Ksq = ki ** 2  # denominator
        # inputs for semivars:
        bjn_nom = np.multiply(dKi, Q)
        bjn_ingredients = np.divide(bjn_nom, Ksq)
        bjn_multiplier = 2 * er
        # semivariance
        moments_mfivu_bjn = bjn_multiplier * np.sum(bjn_ingredients[otmcalls]) / mat
        moments_mfivd_bjn = bjn_multiplier * np.sum(bjn_ingredients[otmputs]) / mat
        # total variance
        moments_mfiv_bjn = moments_mfivu_bjn + moments_mfivd_bjn

        #######################################################################
        ### compute MFIV_BKM - Bakshi, Kapadia and Madan (2003)
        Ksq = ki ** 2
        # inputs for semivars:
        bkm_nom = np.multiply(dKi, np.multiply(1 - np.log(ki), Q))
        bkm_ingredients = np.divide(bkm_nom, Ksq)
        bkm_multiplier = 2 * er
        # semivariance
        moments_mfivu_bkm = bkm_multiplier * np.sum(bkm_ingredients[otmcalls]) / mat
        moments_mfivd_bkm = bkm_multiplier * np.sum(bkm_ingredients[otmputs]) / mat
        # total variance
        moments_mfiv_bkm = moments_mfivu_bkm + moments_mfivd_bkm

        # OUTPUT 1
        dicct.update({'smfiv': moments_smfiv,
                      'mfiv_bkm': moments_mfiv_bkm,
                      'mfiv_bjn': moments_mfiv_bjn})

        if params.get('semivars', {}).get('compute', False):
            dicct.update({'smfivd': moments_smfivd,
                          'mfivd_bkm': moments_mfivd_bkm,
                          'mfivd_bjn': moments_mfivd_bjn})

        #######################################################################
        ### compute some inputs for MFIS/MFIK based on BKM
        a = 2 * (u - 1)
        b1 = 1 - (np.log(1 + k) / m) * ic
        b1.shape = currcalls.shape
        b1 = np.multiply(b1, currcalls)
        h = np.power(u, ic)
        h.shape = currcalls.shape
        b1 = np.divide(b1, h)
        b2 = 1 - (np.log(1 + k) / m) * ip
        b2.shape = currputs.shape
        b2 = np.multiply(b2, currputs)
        g = np.power(u, ip)
        g.shape = currputs.shape
        b2 = np.divide(b2, g)
        b_all = np.concatenate((b2, b1))  # note: puts first because ki has moneyness range from puts to calls

        # if not called just to compute market variance -- proceed with other computations
        # MFIS/ MFIK
        if params.get('mfismfik', {}).get('compute', False):
            mfismfik_dic = compute_skew_kurtosis(m, k, u, ic, ip, currcalls, currputs, er,
                                                 moments_mfiv_bkm * mat / er)  # adjustment /er on 2023-03-30
        # OUTPUT MFIS/ MFIK
        dicct.update(mfismfik_dic)

        #######################################################################
        # CORRIDOR VIX / Bondarenko et al
        if params.get('cvix', {}).get('compute', False):
            cvix_dic = cvix_func(b_all, params['cvix'], ki, moments_mfiv_bkm, mat, a)
            dicct.update(cvix_dic)
            pass

        #######################################################################
        # RIX - Gao Gao Song RFS
        if params.get('rix', {}).get('compute', False):
            rix = moments_mfivd_bkm - moments_mfivd_bjn
            rixn = rix / moments_mfivd_bkm
            dicct.update({'rix': rix, 'rixnorm': rixn})
            pass

        #######################################################################
        # TAIL LOSS MEASURE Hamidieh / Vilkov and Xiao
        if params.get('tlm', {}).get('compute', False):
            if params.get('atmfwd', False):
                x, currputd = Black_delta(1, kiputs, rate, iv[otmcalls == False], mat)
            else:
                x, currputd = BlackScholes_delta(1, kiputs, rate, iv[otmcalls == False], mat)
            tlm_dic = tlm_func(currputs, kiputs, currputd, params['tlm'], moments_mfiv_bkm, mat)
            dicct.update(tlm_dic)
            pass

        #######################################################################
        # SLOPE Kelly et al + applications in Carbon Tail Risk and Pricing Climate Change Exposure
        if params.get('slope', {}).get('compute', False):
            delta_lim = params['slope'].get('deltaP_lim', [-0.5, -0.05])
            ivputs = iv[otmputs]
            if params.get('atmfwd', False):
                x, currputd = Black_delta(1, kiputs, rate, ivputs, mat)
            else:
                x, currputd = BlackScholes_delta(1, kiputs, rate, ivputs, mat)
            selputs = (currputd >= min(delta_lim)) & (currputd <= max(delta_lim))
            dnow = currputd[selputs]
            ivnow = ivputs[selputs]

            if sum(selputs) > 3:
                slope, intercept, r_value, p_value, std_err = ss.linregress(dnow, ivnow)  # x, y ;)
            else:
                slope = np.nan
                pass
            dicct.update({'slopedn': slope})

            # same for calls
            delta_lim = params['slope'].get('deltaC_lim', [0.05, 0.5])
            ivcalls = iv[otmcalls]
            if params.get('atmfwd', False):
                currcalld, x = Black_delta(1, kicalls, rate, ivcalls, mat)
            else:
                currcalld, x = BlackScholes_delta(1, kicalls, rate, ivcalls, mat)
            selcalls = (currcalld >= min(delta_lim)) & (currcalld <= max(delta_lim))
            dnow = currcalld[selcalls]
            ivnow = ivcalls[selcalls]

            if sum(selcalls) > 3:
                slope, intercept, r_value, p_value, std_err = ss.linregress(dnow, ivnow)  # x, y ;)
            else:
                slope = np.nan
                pass
            dicct.update({'slopeup': -slope})
            pass

        #######################################################################
        out = dicct
        if output == 'pandas':
            out = pd.Series(dicct)
            pass
    return out


### compute MFIS/MFIK
def compute_skew_kurtosis(m, k, u, ic, ip, currcalls, currputs, er, V):
    u = u.ravel()
    ic = ic.ravel()
    ip = ip.ravel()
    currcalls = currcalls.ravel()
    currputs = currputs.ravel()

    result_dict = {}

    a = 3 * (u - 1) * np.log(1 + k) / m
    b1 = np.sum(ic * (2 - (np.log(1 + k) / m) * ic) * currcalls / u ** ic)
    b2 = np.sum(ip * (2 - (np.log(1 + k) / m) * ip) * currputs / u ** ip)
    W = a * (b1 + b2)

    a = 4 * (u - 1) * (np.log(1 + k) / m) ** 2
    b1 = np.sum(ic ** 2 * (3 - (np.log(1 + k) / m) * ic) * currcalls / u ** ic)
    b2 = np.sum(ip ** 2 * (3 - (np.log(1 + k) / m) * ip) * currputs / u ** ip)
    X = a * (b1 + b2)

    mu = er - 1 - er / 2 * V - er / 6 * W - er / 24 * X
    c = (er * V - mu ** 2)

    mfis = (er * W - 3 * mu * er * V + 2 * mu ** 3) / (c ** (3 / 2))
    mfik = (er * X - 4 * mu * er * W + 6 * er * mu ** 2 * V - 3 * mu ** 4) / c ** 2

    result_dict['mfis'] = mfis.item()
    result_dict['mfik'] = mfik.item()

    return result_dict


### compute TAIL LOSS MEASURE
def tlm_func(currputs, kiputs, currputd, tlm_params, iv, mat):
    currputs = currputs.ravel()
    kiputs = kiputs.ravel()
    currputd = currputd.ravel()
    result_dict = {}

    sd_ls = tlm_params.get('vol_lim', [])
    absD_ls = tlm_params.get('delta_lim', [])

    # define time varying threshold
    iv_m = np.sqrt(iv * mat)  # convert to horizon = days
    for r in sd_ls:
        tlm = np.NAN
        k0 = 1.0 - r * iv_m
        goods = kiputs <= k0
        if np.sum(goods) > 1:
            Pt = currputs[goods]
            K = kiputs[goods]  # strikes
            P0 = np.max(Pt)
            ind = np.argmax(Pt)
            K0 = K[ind]
            try:
                res = minimize(f_tlm, np.array([2, 0.01]), args=(P0, K0, Pt, K,),
                               method='SLSQP',
                               options={'ftol': 1e-32, 'disp': False})
                tlm = res.x[1] / (1 - res.x[0])
            except:
                pass
            pass
        result_dict['tlm_sigma' + str(r)] = tlm
        pass

    for r in absD_ls:
        tlm = np.NAN
        if abs(r) > 1: r = r / 100
        goods = abs(currputd) <= abs(r)
        if np.sum(goods) > 1:
            Pt = currputs[goods]  # prices
            K = kiputs[goods]  # strikes
            P0 = np.max(Pt)
            ind = np.argmax(Pt)
            K0 = K[ind]
            try:
                res = minimize(f_tlm, np.array([2, 0.01]), args=(P0, K0, Pt, K,),
                               method='SLSQP', options={'ftol': 1e-32, 'disp': False})
                tlm = res.x[1] / (1 - res.x[0])
            except:
                pass
            pass
        result_dict['tlm_delta' + str(int(abs(r * 100)))] = tlm
        pass
    return result_dict


###############################################################################
def f_tlm(X, P0, K0, Pt, K):
    # prices = Pt
    # strikes = K
    xi = X[0]
    beta = X[1]
    checkterms = 1 + (xi / beta) * (K0 - K)
    term2 = np.multiply(P0, np.divide(np.power(checkterms, (1 - 1 / xi)), Pt))
    return np.sum((1 - term2) ** 2)


### compute CORRIDOR VIX
def cvix_func(b_all, cvix_params, ki, iv, mat, a):
    '''
    Computes corridor variance based on r relative (sigma) deviations
    or specified absolute deviations from ATM moneyness of 1
    '''
    # change a to adjust for maturity
    a = a / mat

    sd_ls = cvix_params.get('vol_dev', [])
    absD_ls = cvix_params.get('abs_dev', [])

    result_dict = {}

    # define time varying threshold for semi-variances
    iv_m = np.sqrt(iv * mat)  # convert to horizon = days

    b_all.shape = ki.shape

    # compute CVIX based on r standard deviations (mfiv**0.5)
    for r in sd_ls:
        kl = 1 - r * iv_m
        ku = 1 + r * iv_m
        sel_ki = (ki >= kl) & (ki <= ku)  # range for cvix
        result_dict['cvix_sigma' + str(r)] = a * np.sum(b_all[sel_ki])

    # compute CVIX based on absolute deviation
    for r in absD_ls:
        kl = 1 - r
        ku = 1 + r
        sel_ki = (ki >= kl) & (ki <= ku)  # range for CVIX
        result_dict['cvix_mnes' + str(int(r * 100))] = a * np.sum(b_all[sel_ki])

    return result_dict


# *****************************************************************************
# COMPUTE BS AND BS DELTA
# *****************************************************************************
def BlackScholes(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + np.square(sigma) / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - np.square(sigma) / 2) * T) / (sigma * np.sqrt(T))
    c = S0 * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
    p = K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * ss.norm.cdf(-d1)
    return c, p


def Black(F, K, r, sigma, T):
    d1 = (np.log(F / K) + (np.square(sigma) / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) + (-np.square(sigma) / 2) * T) / (sigma * np.sqrt(T))
    c = (F * ss.norm.cdf(d1) - K * ss.norm.cdf(d2)) * np.exp(-r * T)
    p = (K * ss.norm.cdf(-d2) - F * ss.norm.cdf(-d1)) * np.exp(-r * T)
    return c, p


def BlackScholes_delta(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + np.square(sigma) / 2) * T) / (sigma * np.sqrt(T))
    delta_c = ss.norm.cdf(d1)
    delta_p = delta_c - 1
    return delta_c, delta_p


def Black_delta(F, K, r, sigma, T):
    d1 = (np.log(F / K) + (np.square(sigma) / 2) * T) / (sigma * np.sqrt(T))
    delta_c = ss.norm.cdf(d1) * np.exp(-r * T)
    delta_p = -ss.norm.cdf(-d1) * np.exp(-r * T)
    return delta_c, delta_p


# *****************************************************************************
# % Function OM_Interpolate_IV
# *****************************************************************************
def interpolate_iv_by_moneyness(mnes, vol, grid):
    # set the grid in terms of moneyness that we want to use to compute the
    # MFIV/ MFIS and other integrals as needed
    m = grid[0]  # use points -500:500, i.e. 1001 points for integral approximation
    k = grid[1]  # use moneyness 1/(1+2) to 1+2 - should be enough as claimed by JT
    u = np.power((1 + k), (1 / m))

    # create strikes using s=1, i.e. get k/s = moneyness
    mi = np.arange(-m, m + 1)
    ki = np.power(u, mi)
    iv = np.empty((len(ki), 1))
    iv[:] = np.NAN
    ki.shape = iv.shape

    currspline = PchipInterpolator(mnes, vol, axis=0)

    k_s_max = max(mnes)  # OTM calls
    k_s_min = min(mnes)  # OTM puts
    iv_max = vol[0]  # for more OTM puts i.e we have iv_max for min k/s, i.e. for OTM put option
    iv_min = vol[-1]  # for more OTM calls  i.e. we have iv_min for OTM call option

    # calculate the interpolated/ extrapolated IV for these k/s
    ks_larger_ind = ki > k_s_max  # more OTM call
    ks_smaller_ind = ki < k_s_min  # more OTM put
    ks_between_ind = (ki >= k_s_min) & (ki <= k_s_max)

    if sum(ks_larger_ind) > 0:
        iv[ks_larger_ind] = iv_min

    if sum(ks_smaller_ind) > 0:
        iv[ks_smaller_ind] = iv_max

    # evaluate the spline at ki[ks_between_ind]
    if sum(ks_between_ind) > 0:
        s = currspline(ki[ks_between_ind], nu=0, extrapolate=None)
        s.shape = iv[ks_between_ind].shape
        iv[ks_between_ind] = s
    return iv, ki
