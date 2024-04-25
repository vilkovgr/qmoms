import pandas as pd
from qmoms import default_surf_dtype_mapping, default_rate_dtype_mapping, default_date_format
from importlib.resources import files


def load_data(surf_dtype_mapping=default_surf_dtype_mapping,
              rate_dtype_mapping=default_rate_dtype_mapping,
              surf_date_format=default_date_format,
              rate_date_format=default_date_format,
              rate_factor=0.01):
    """
    Load and process example datasets for surface and zero coupon rate data from CSV files.

    Parameters:
    - surf_dtype_mapping (dict, optional): Data types for surface data columns. Defaults to:
                                            {'id': 'int64', 'date': 'str', 'days': 'int64', 'mnes': 'float64',
                                            'prem': 'float64', 'impl_volatility': 'float64', 'delta': 'float64'}.
    - rate_dtype_mapping (dict, optional): Data types for zero coupon rate data columns.
                                            Defaults to: {'date': 'str', 'days': 'int64', 'rate': 'float64'}
    - surf_date_format (str, optional): Date format for surface data, used to parse string dates into datetime objects.
    - rate_date_format (str, optional): Date format for zero coupon rate data, similar to surf_date_format.
    - rate_factor (float, optional): Factor by which the 'rate' column values are adjusted. Defaults to 0.01.

    Returns:
    - tuple(pandas.DataFrame, pandas.DataFrame): Returns two DataFrames:
        - df_surf: Contains surface data, where 'delta' is adjusted and 'date' is converted.
        - df_rate: Contains zero coupon rate data, where 'rate' is adjusted by 'rate_factor' and 'date' is converted.

    Examples:
    ```python
    df_surf, df_rate = load_data()
    ```
    """
    data_files = files('data')
    df_surf = pd.read_csv(data_files / 'surface.csv', sep=',', dtype=surf_dtype_mapping)
    df_surf['date'] = pd.to_datetime(df_surf['date'], format=surf_date_format, errors='coerce')
    df_surf['delta'] = df_surf['delta'] / 100

    df_rate = pd.read_csv(data_files / 'zerocd.csv', sep=',', dtype=rate_dtype_mapping)
    df_rate['date'] = pd.to_datetime(df_rate['date'], format=rate_date_format, errors='coerce')
    df_rate['rate'] = df_rate['rate'] * rate_factor

    return df_surf, df_rate

