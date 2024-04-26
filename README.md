# Implied Moments from Volatility Surface Data

This is a Python implementation of a set of option-implied characteristics.

## Scope
The package contains functions to compute option-implied moments and characteristics from implied volatility surface data. 
The computations are based on the out-the-money (OTM) implied volatilities, interpolated as function of moneyness 
(Strike/Underlying Price or Strike/ Forward Price) within the avaialable moneyness range and set to the boundary values 
outside to fill in the pre-specified moneyness range. OTM is defined as moenyess>=1 for calls and <1 for puts. 

The following moments and characteristics are computed: 
1. Model-free implied variance (log contract):
   * `MFIV_BKM` using Bakshi, Kapadia, and Madan (RFS, 2003) / https://doi.org/10.1093/rfs/16.1.0101
   * `MFIV_BJN` using Britten-Jones and Neuberger (JF, 2002) / https://doi.org/10.1111/0022-1082.00228
2. Simple model-free implied variance 
   * `SMFIV` using Martin (QJE, 2017) / https://doi.org/10.1093/qje/qjw034
3. Corridor VIX 
   * `CVIX` using Andersen, Bondarenko, and Gonzalez-Perez (RFS, 2015) / https://doi.org/10.1093/rfs/hhv033
4. Tail Loss Measure 
   * `TLM` using Vilkov, Xiao (2012) / https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1940117 and Hamidieh (Journal of Risk, 2017) /
     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2209654
5. Semivariances (only down) for log and simple contracts 
    * Following Feunou, Jahan-Parvar,and Okou, (Journal of Fin Econometrics, 2017) / https://doi.org/10.1093/jjfinec/nbx020
    * Up semivariances can be computed as the respective total variance minus down semivariance
6. Tail jump measure 
   * `RIX` using Gao, Gao and Song, (RFS, 2018) / https://doi.org/10.1093/rfs/hhy027
7. Ad hoc tail measures (smile steepness for the tails) 
   * `Slopeup` (right tail) and `Slopedn` (left tail) measure right and left tail slope, respectively. Larger values indicate 
   more expensive tail relative to the at-the-money level. 
   * Used in Carbon Tail Risk, Ilhan, Sautner, and Vilkov, (RFS, 2021) / https://doi.org/10.1093/rfs/hhaa071 and
   Pricing Climate Change Exposure, v.Lent, Sautner, Zhang, and Vilkov, (Management Science, 2023) / https://doi.org/10.1287/mnsc.2023.4686
   (mind the sign of slopeup -- we take '-Slope' here for better interpretability). 
   Originally, Slopedn is used in Kelly, Pastor, Veronesi (JF, 2016) / https://doi.org/10.1111/jofi.12406 

## Usage

Exemplary use of the `qmoms` package. The data is the provided with the package, and are composed of two files: 

1. `surface.csv`: the implied volatility surface as a function of option delta for several dates in 1996 and 30-day maturity.
The columns mapping is specified in `qmoms_params.py` and can be imported from the main package:
```python
from qmoms import default_surf_dtype_mapping
default_surf_dtype_mapping = {'id': 'int64', 
                              'date': 'str', 
                              'days': 'int64', 
                              'mnes': 'float64', 
                              'prem': 'float64', 
                              'impl_volatility': 'float64', 
                              'delta': 'float64'}
```
2. `zerocd.csv`: the zero CD rates recorded on each date for several maturities, in % p.a.
```python
from qmoms import default_rate_dtype_mapping
default_rate_dtype_mapping = {'date': 'str',  
                              'days': 'int64',
                              'rate': 'float64'}
```
The default date format for these files is provided in `qmoms_params.py`: `default_date_format = '%d%b%Y'` 

To load the data, use the provided function `load_data()`
```python
from qmoms.examples import load_data
df_surf, df_rate = load_data()
```


There are two main functions provided: `qmoms_compute()` and `qmoms_compute_bygroup()`:

The first function computes a set of variables as specified in `params` dictionary for one instance of the surface 
(i.e., one id/date/days combination).
```python
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
```

The second function `qmoms_compute_bygroup` computes the variables for a set of instances collected in a DataFrame.
Note that the input variable `groupparams` shall be a tuple or a list with the 0th element containing the 
group (dataframe) and the 1st element being a dictionary with parameters. Asset `id`, current `date`, 
maturity in `days`, and current `rate` in decimals p.a. can be provided as individual variables, or in the 
group DataFrame. If these variables are provided directly to the function, they will take precedence over any values 
provided within the DataFrame. 
The DataFrame column names shall conform to the `cols_map` mapping.

```python
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
    group = group.sort_values(by=['mnes'])
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
```

The usage of both functions is illustrated below: 

```python
'''the script with examples is provided in qmoms/examples/qmoms_test.py'''
# import packages and variables
import pandas as pd
from tqdm import *
from multiprocessing import cpu_count, Pool
from qmoms.examples import load_data
from qmoms import default_params, default_surf_dtype_mapping, filter_options, get_rate_for_maturity
from qmoms import qmoms_compute, qmoms_compute_bygroup

# set number of used cores for parallel computation
if cpu_count() > 30:
    CPUUsed = 20
else:
    CPUUsed = cpu_count()

if __name__ == '__main__':
    # load the provided data 
    df_surf, df_rate = load_data()

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
```


## Installation 

You can install the package master branch directly from GitHub with:

```bash
pip install git+https://github.com/vilkovgr/qmoms.git
```


## Requirements

* Python 3.9+
* numpy>=1.19 
* pandas>=2.0 
* scipy>=1.10 
* tqdm>=4.0

Older versions might work, but were not tested. 

## Other Channels and Additional Data
Precomputed moments and other data are available through the OSF repository at https://osf.io/8awyu/  

Feel free to load and use in your projects:  
* Generalized lower bounds as in Chabi-Yo, Dim, Vilkov (MS, 2023) / https://ssrn.com/abstract=3565130 
* Option-implied betas as in Buss and Vilkov (RFS, 2012) / https://ssrn.com/abstract=1301437 
* Options-implied correlations as in Driessen, Maenhout, Vilkov (JF, 2009) / https://ssrn.com/abstract=2166829 
* Climate change exposure measures as in Lent, Sautner, Zhang, and Vilkov (JF, MS, 2023) / https://ssrn.com/abstract=3792366 
and https://ssrn.com/abstract=3642508 
 

## Acknowledgements

The implementation is inspired by the Python and MATLAB code for implied moments made available on [Grigory Vilkov's website](https://vilkov.net).

***
The package is still in the development phase, hence please share your comments and suggestions with us.

Contributions welcome!

Grigory Vilkov