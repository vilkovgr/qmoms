# import packages and variables
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import warnings

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# *****************************************************************************
# execution over grouped data -- parallel and serial
# *****************************************************************************
def applyParallel(dfGrouped, func_main, params, CPUUsed=20):
    with Pool(CPUUsed) as p:
        ret_list = p.map(func_main, [[group_indv, params] for name, group_indv in dfGrouped])
        out = list(filter(None.__ne__, ret_list))
        out = pd.DataFrame.from_records(out)
    return out


def applySerial(dfGrouped, func_main, params):
    ret_list = [func_main([group_indv, params]) for name, group_indv in dfGrouped]
    out = list(filter(None.__ne__, ret_list))
    out = pd.DataFrame.from_records(out)
    return out


# *****************************************************************************
# % Filter Dataframes, apply this for a whole dataset and a specific grid
# *****************************************************************************
def filter_options(optdata, filter):
    g1 = (optdata['mnes'] >= filter['mnes_lim'][0]) & (optdata['mnes'] <= filter['mnes_lim'][1])
    g2 = (optdata['delta'] >= filter['delta_call_lim'][0]) & (optdata['delta'] <= filter['delta_call_lim'][1])
    g3 = (optdata['delta'] >= filter['delta_put_lim'][0]) & (optdata['delta'] <= filter['delta_put_lim'][1])
    g1 = g1 & (g2 | g3)

    if 'open_interest' in optdata.columns:
        g1 = g1 & (optdata['open_interest'] >= filter['open_int_zero'])
    if ('best_bid' in optdata.columns):
        g1 = g1 & (optdata['best_bid'] >= filter['best_bid_zero'])
        if ('best_offer' in optdata.columns):
            g1 = g1 & ((optdata['best_bid'] + optdata['best_offer']) / 2 >= filter['min_price'])

    # moneyness is sorted now from low to high
    optdata4interp = optdata.loc[g1, :].sort_values(['id', 'date', 'mnes'], ascending=[True, True, True]).copy()
    return optdata4interp


# *****************************************************************************
# get rate with linear interpolation / pass ratedata as input
# *****************************************************************************
def get_rate_for_maturity(df_rate, df_surf=None, date=None, days=None):
    """
    Retrieves or interpolates the interest rate for a given date and maturity or across a surface of dates and maturities.

    Parameters:
    - df_rate (pd.DataFrame): DataFrame containing interest rates with columns 'date', 'days', and 'rate'.
    - df_surf (pd.DataFrame, optional): DataFrame representing the surface data with 'date' and 'days' columns.
    If provided, rates are interpolated for the entire surface.
    - date (str, optional): The specific date for which the rate needs to be retrieved or interpolated.
    Required if df_surf is not provided.
    - days (int, optional): The specific number of days until maturity for which the rate is required.
    Required if df_surf is not provided.

    Returns:
    - float or pd.DataFrame: Returns a single interpolated rate if 'date' and 'days' are provided.
    Returns a DataFrame with interpolated rates across the surface if 'df_surf' is provided.

    The function first checks if the input is sufficient to perform the operation. It then either:
    1. Retrieves and interpolates the rate for a single date and maturity when 'date' and 'days' are provided.
    2. Interpolates rates for an entire surface provided by 'df_surf' DataFrame, updating the 'rate' column based
    on available data in 'df_rate'.

    Errors:
    - Prints an error and returns NaN if neither sufficient single date and days nor a DataFrame surface is provided.

    Example:
    ```python
    # Single rate interpolation
    rate_on_date = get_rate_for_maturity(df_rate, date='2023-01-01', days=360)

    # Surface rate interpolation
    df_interpolated_rates = get_rate_for_maturity(df_rate, df_surf=df_surface)
    ```
    """

    if (df_surf is None) and ((date is None) or (days is None)):
        print('Error: Not enough inputs, provide date and days')
        return np.nan

    # return just a number for that particular date
    if ((date is not None) and (days is not None)):
        # get rate for first_date
        ZeroCoupon = df_rate[df_rate['date'] == date].copy()
        # if date is not available, get last available date
        if ZeroCoupon.empty:
            ZeroCoupon = df_rate[df_rate['date'] <= date].copy()
            last_available_date = ZeroCoupon['date'].iloc[-1]
            ZeroCoupon = df_rate[df_rate['date'] == last_available_date].copy()
            pass
            # get the rate with the maturity
        ZeroCoupon.sort_values(by=['date', 'days'], inplace=True)
        x = days
        xp = ZeroCoupon['days']
        fp = ZeroCoupon['rate']
        y = np.interp(x, xp, fp, left=None, right=None)
        return y

    # interpolate for the whole surface and return a frame
    if isinstance(df_surf, pd.DataFrame):
        if 'rate' in df_surf.columns:
            df_surf = df_surf.drop(columns=['rate'])
        df_surf_rate = df_surf[['date', 'days']].drop_duplicates()
        df_surf_rate['rate'] = np.nan

        # Merge df_rate and df_surf_rate on 'date'
        merged_df = pd.concat([df_rate, df_surf_rate], axis=0)

        # Sort by 'date' and 'days_surf' because we want to interpolate based on df_surf_rate's 'days'
        merged_df.sort_values(by=['date', 'days'], inplace=True)

        # Now group by 'date' and apply the interpolation for each group in the 'days' dimension
        def interpolate_group(group):
            # Ensure the days are sorted for interpolation
            group = group.sort_values(by='days')
            bads = group['rate'].isna()
            x = group.loc[bads, 'days']
            xp = group.loc[~bads, 'days']
            fp = group.loc[~bads, 'rate']
            y = np.interp(x, xp, fp, left=None, right=None)
            # Interpolate the missing rates
            group.loc[bads, 'rate'] = y
            return group

        # Apply interpolation to each group
        interpolated_df = merged_df.groupby(['date'])[['days', 'rate']]. \
            apply(interpolate_group). \
            reset_index()[['date', 'days', 'rate']]

        df_surf = df_surf.merge(interpolated_df, how='left', on=['date', 'days'])
        return df_surf
