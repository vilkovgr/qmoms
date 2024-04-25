# -*- coding: utf-8 -*-
"""
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
_________________________________________________________________________________
Partially based on Matlab code for implied moments by G.Vilkov (www.vilkov.net)
Original math is from
1. Bakshi, Kapadia, and Madan (RFS, 2003) for MFIV_BKM + Britten-Jones and Neuberger (JF, 2002) for MFIV_BJN
https://doi.org/10.1093/rfs/16.1.0101 and  https://doi.org/10.1111/0022-1082.00228
2. Martin (QJE, 2017) for SMFIV
https://doi.org/10.1093/qje/qjw034
3. Andersen, Bondarenko, and Gonzalez-Perez (RFS, 2015) for CVIX
https://doi.org/10.1093/rfs/hhv033
4. Vilkov, Xiao (2012) and Hamidieh (Journal of Risk, 2017) for TLM
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1940117 and https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2209654
5. Semivariances are based on CVIX derivation from #3/
following Feunou, Jahan-Parvar,and Okou, (Journal of Fin Econometrics, 2017)
https://doi.org/10.1093/jjfinec/nbx020
6. RIX following Gao, Gao and Song, (RFS, 2018)
https://doi.org/10.1093/rfs/hhy027
7. Slopeup and Slopedn measures as used in Carbon Tail Risk, Ilhan, Sautner, and Vilkov, (RFS, 2021) and
Pricing Climate Change Exposure, v.Lent, Sautner, Zhang, and Vilkov, (Management Science, 2023)
https://doi.org/10.1093/rfs/hhaa071 and https://doi.org/10.1287/mnsc.2023.4686
(mind the sign of slopeup -- we take '-Slope' here for better interpretability)
Originally Slopedn is used in https://doi.org/10.1111/jofi.12406 Kelly, Pastor, Veronesi (JF, 2016)

Data required:
1. Zero-cd rates
-- one can use zero cd rates from OptionMetrics, or any other provider, e.g., short-term t-bill rates from FRED

2. Surface data from OptionMetrics or any other provider
-- see attached files for 1996 for all SP500 components and SP itself (ID = 99991) 
    for formats and names of the columns
-- note that the ID gives the PERMNO of the company and is based on CRSP data

Output: csv file with all the required option-based measures (the file with specified measures for 1996 is attached)

Notes:
-- the code is written in a very simple form for a purpose -- this way it is easier to debug and adopt;
Please write me with any comments on vilkov@vilkov.net

"""
import numpy as np
# %% Load Variables and Dataframes
# import packages and variables
import pandas as pd
from tqdm import *
from multiprocessing import cpu_count, Pool
from qmoms.examples import load_data
from qmoms import default_params, default_surf_dtype_mapping, filter_options, get_rate_for_maturity
from qmoms import qmoms_compute, qmoms_compute_bygroup

# DEFAULT PARAMETERS FOR THE PROCEDURE
'''
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
'''
##########
'''
to modify parameters, update the default_params dictionary with the dictionary with desired parameters
for example, in case you do not want to compute the tlm, and want to change slope parameters, use
new_params0 = {'tlm': {'compute': False}}
new_params1 = {'slope': {'compute': True,
                            'deltaP_lim': [-0.4, -0.1],
                            # limits to compute the slopedn as Slope from  IV = const + Slope * Delta
                            'deltaC_lim': [0.05, 0.5]
                            # limits to compute the slopeup as -Slope from IV = const + Slope * Delta
                            }}
params = default_params | new_params0 | new_params1
'''

# set number of used cores for parallel computation
if cpu_count() > 30:
    CPUUsed = 20
else:
    CPUUsed = cpu_count()

# %% ALL FUNCTIONS THAT ARE USED IN COMPUTATIONS ARE IMPORTED FROM fn_defs_qmoments
# *****************************************************************************
# *****************************************************************************
if __name__ == '__main__':

    df_surf, df_rate = load_data()
    df_rate['rate'] = df_rate['rate']  # rates in decimals p.a.

    #######################################################################
    # FILTER THE DATA USING DEFAULT PARAMS
    # note that you can easily define your own filter -- make sure you select
    # only OTM options with non-repeating moneyness levels
    # we use OTM calls identified by mnes >=1

    # select only relevant columns and order them
    df_surf = df_surf[default_surf_dtype_mapping.keys()]
    # Filter the whole dataset
    df_surf = filter_options(df_surf, default_params['filter'])
    print("loading and filter done")
    #######################################################################

    #######################################################################
    # GROUP THE DATA FOR COMPUTATIONS, AND TEST THE OUTPUT ON ONE GROUP
    df_surf_grouped = df_surf.groupby(['id', 'date', 'days'], group_keys=False)
    # create an iterator object
    group = iter(df_surf_grouped)
    # select the next group
    group_next = next(group)
    ids, group_now = group_next
    # extract group identifiers
    id, date, days = ids
    # extract data on moneyness and implied volatility
    mnes = group_now.mnes
    iv = group_now.impl_volatility
    # interpolate the rate to the exact maturity
    rate = get_rate_for_maturity(df_rate, date=date, days=days)
    # feed to the function computing implied moments
    qout = qmoms_compute(mnes=mnes, vol=iv, days=days, rate=rate, params=default_params)
    print(ids, qout)

    #######################################################################
    # GROUP THE DATA, AND RUN COMPUTATIONS IN A LOOP SEQUENTIALLY
    qmoms_all = {}
    df_surf_grouped = df_surf.groupby(['id', 'date', 'days'], group_keys=False)
    for ids, group_now in tqdm(df_surf_grouped):
        id, date, days = ids
        # extract data on moneyness and implied volatility
        mnes = group_now.mnes
        iv = group_now.impl_volatility
        # interpolate the rate to the exact maturity
        rate = get_rate_for_maturity(df_rate, date=date, days=days)
        # feed to the function computing implied moments
        qout = qmoms_compute(mnes=mnes, vol=iv, days=days, rate=rate, params=default_params)
        qmoms_all[ids] = qout
        pass

    qmoms_all = pd.DataFrame(qmoms_all).T
    qmoms_all.index.names = ['id', 'date', 'days']
    print(qmoms_all)

    #######################################################################
    # GROUP THE DATA, AND RUN COMPUTATIONS FOR THE WHOLE DATAFRAME
    # first, add the interest rate for each maturity to the surface dataframe
    df_surf = get_rate_for_maturity(df_rate, df_surf=df_surf)
    # group dataset by id,date,days
    grouped = df_surf.groupby(['id', 'date', 'days'], group_keys=False)

    # PARALLEL
    with Pool(CPUUsed) as p:
        ret_list = p.map(qmoms_compute_bygroup, [[group_indv, default_params] for name, group_indv in grouped])
        out_p = pd.DataFrame.from_records(ret_list)
        pass
    print(out_p)

    # SERIAL
    ret_list = [qmoms_compute_bygroup([group_indv, default_params]) for name, group_indv in grouped]
    out_s = pd.DataFrame.from_records(ret_list)
    print(out_s)

