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

# MAIN PARAMETERS FOR THE PROCEDURE
default_params = {'atmfwd': False,  # if True, use FWD as ATM level and Black model for option valuation
                  'grid': {'number_points': 500,  # points*2 + 1
                           'grid_limit': 2},  # 1/(1+k) to 1+k; typically, k = 2
                  'filter': {'mnes_lim': [0, 1000],  # [0.7, 1.3],
                             'delta_put_lim': [-0.5 + 1e-3, 0],  # typically, only OTM
                             'delta_call_lim': [0, 0.5],  # typically, only OTM
                             'best_bid_zero': -1,  # if negative, no restriction on zero bid; set to 0 for limit
                             'open_int_zero': -1,  # if negative, no restriction on zero open interest; set to 0 for limit
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

# specify mapping for the file with surface data
default_surf_dtype_mapping = {
                        'id': 'int64',
                        'date': 'str',  # Treat date as a string initially
                        'days': 'int64',
                        'mnes': 'float64',
                        'prem': 'float64',
                        'impl_volatility': 'float64',
                        'delta': 'float64'
                        }

default_rate_dtype_mapping = {
                        'date': 'str',  # Treat date as a string initially
                        'days': 'int64',
                        'rate': 'float64'
                        }

default_date_format = '%d%b%Y'