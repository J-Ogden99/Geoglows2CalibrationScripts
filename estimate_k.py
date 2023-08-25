import os
from datetime import datetime
from glob import glob
import requests

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from simpledbf import Dbf5
from scipy import stats as st
from scipy.optimize import curve_fit
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, PolynomialFeatures

from matchPoints import match_rivers_to_gauges
from getVelocities import get_values_from_usgs, df_from_tab_separated


def is_numeric(column: pd.Series):
    """
    Checks if a pandas series is numeric
    """
    try:
        pd.to_numeric(column)
        return True
    except ValueError:
        return False


def insert_substring(original_string, index, substring):
    return original_string[:index] + substring + original_string[index:]


def dms_to_decimal(degrees, minutes, seconds):
    return float(degrees) + (float(minutes) / 60) + (float(seconds) / 3600)


def string_to_lat_lon(val):
    if '.' in val:
        val = ''.join(val.split('.')[:-1])
    for i in (-2, -5):
        val = insert_substring(val, i, '.')

    split = val.split('.')

    return dms_to_decimal(*split)


def remove_outliers(obj, col=None, outlier_threshold: int = 3):
    """
    Removes all outliers column-by-column from a pandas dataframe or numpy array

    Args:
       obj: A pandas dataframe or numpy array
       col: A column name (for pandas) or column index (for numpy). If given, only the specified column
            will be checked for outliers
       outlier_threshold: z-score threshold that will be used to determine whether a value is an outlier
            or not

    Returns:
        A copy of the obj with the outliers removed

    """
    if isinstance(obj, pd.DataFrame):
        if col:
            z_scores = st.zscore(obj[col])
            new_df = obj[~(np.abs(z_scores) > outlier_threshold)]
            return new_df
        for col in obj.columns:
            if not is_numeric(obj[col]):
                continue
            # mask = np.zeros(obj.shape[0], dtype=bool)
            # while not mask.all():
            z_scores = st.zscore(obj[col])
            mask = ~(np.abs(z_scores) > outlier_threshold)
            obj = obj[mask]
        return obj
    elif isinstance(obj, np.ndarray):
        if col:
            z_scores = st.zscore(obj[:, col])
            new_arr = obj[~(np.abs(z_scores) > outlier_threshold)]
            return new_arr
        for col in range(obj.shape[1]):
            # mask = np.zeros(obj.shape[0], dtype=bool)
            # while not mask.all():
            z_scores = st.zscore(obj[:, col])
            mask = ~(np.abs(z_scores) > outlier_threshold)
            obj = obj[mask]
        return obj


def exponential_func(x, a, b):
    return a * np.exp(b * x)


def polynomial_func(x, a, b, c, d, e):
    return a * x**(1/b) + c * x**d + e

def sqrt_func(x, a, b):
    return a * x**(1/b)


def mannings(n, r, s):
    return (1.49/n) * r ** (2/3) * s ** (1/2)

# Options for printouts
pd.set_option('display.max_columns', 20)
np.set_printoptions(suppress=True)

# Decide whether to read nhd data
read_nhd = False

# Parameters for the retrieval of velocity and discharge values
base_url = f'https://waterdata.usgs.gov/nwis/uv/?referred_module=sw&format=rdb_station_file&group_key=NONE' \
           f'&date_format=YYYY-MM-DD&rdb_compression=file' \
           f'&list_of_search_criteria=site_tp_cd%2Crealtime_parameter_selection'

data_base_url = f'https://waterservices.usgs.gov/nwis/iv/'

velocities_dir = 'Velocities'
discharges_dir = 'Discharges'
rating_curves_dir = 'RatingCurves'
stream_widths_dir = 'StreamWidths'
n = 0.06

fps_velocities = []
mps_velocities = []
for file in os.listdir(velocities_dir):
    file = os.path.join(velocities_dir, file)
    if 'mph' in file:
        df = pd.read_csv(file, index_col=0)
        data_col = df.columns[4]
        df[data_col] = df[data_col] * 1.4667
        new_file = os.path.join(os.path.split(file)[0], os.path.basename(file).replace('mph', 'fps'))
        df.to_csv(new_file)
        fps_velocities.append(new_file)
        continue
    if 'fps' in file:
        fps_velocities.append(file)
        continue
    if 'mps' in file:
        mps_velocities.append(file)

velocity_ids_fps = [os.path.splitext(os.path.basename(file))[0].split('_')[0] for file in fps_velocities]

# Investigate Rating Curves
rating_curves_ids = []
exp_or_poly = []
# for file in glob(os.path.join(rating_curves_dir, f'*.csv')):
#     df = pd.read_csv(file, index_col=0)
#     rating_curves_ids.append(os.path.splitext(os.path.basename(file))[0])
#     remove_outliers(df)
#     if 'Gage Height (ft)' not in df.columns:
#         df = df.rename(columns={'INDEP': 'Gage Height (ft)', 'DEP':'Discharge (ft^3/s)'})
#     y = df['Gage Height (ft)']
#     x = df['Discharge (ft^3/s)']
#     model = make_pipeline(PolynomialFeatures(include_bias=False), linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13)))
#     # model.fit(np.sqrt(x.to_numpy()).reshape(-1, 1), y)
#     # r2 = model.score(x)
#
#     # x = x / x.max()
#     # y = y / y.max()
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#     axes[0].plot(x, y)
#
#     try:
#         polynomial_params, polynomial_covariance = curve_fit(polynomial_func, x, y, maxfev=50000)
#         axes[1].plot(x, polynomial_func(x, *polynomial_params))
#     except:
#         polynomial_params, polynomial_covariance = curve_fit(sqrt_func, x, y)
#         axes[1].plot(x, sqrt_func(x, *polynomial_params))
#
#     plt.show()
#     print(polynomial_params)

intersection = set(velocity_ids_fps).intersection(set(rating_curves_ids))

# Estimate width (b), slope, or velocity in general based on upstream area, stream order?
st_tps = ['ST', 'ST-CA']
q_pm_cds = ['00060', '00061']
stream_width_cds = ['00004']
if not os.path.exists(discharges_dir):
    get_values_from_usgs(st_tps, q_pm_cds, datetime(2023, 6, 8), datetime(2023, 6, 15), base_url, data_base_url,
                         discharges_dir, filter_ids=velocity_ids_fps)
if not os.path.exists(stream_widths_dir):
    get_values_from_usgs(st_tps, stream_width_cds, datetime(2023, 6, 8), datetime(2023, 6, 15), base_url, data_base_url,
                         stream_widths_dir, filter_ids=velocity_ids_fps)

discharges_df = pd.DataFrame()
velocities_df = pd.DataFrame()
for file in glob(os.path.join(discharges_dir, '*.csv')):
    df = pd.read_csv(file, index_col=0, dtype={'site_no': str})
    df = df.rename(columns={df.columns[4]: 'q'})[['datetime', 'site_no', 'site_name', 'q']]
    df['q'] = pd.to_numeric(df['q'], errors='coerce')
    if not df['q'].notna().all():
        print(f'File {file} contained improper values')
        continue
    if (df['q'] < 0).any().any():
        df.loc[df['q'] < 0, 'q'] = 0.
    if discharges_df.empty:
        discharges_df = df
    else:
        discharges_df = pd.concat([discharges_df, df], axis=0)

# Make a summary table for discharges
q_stats = discharges_df.groupby(['site_no', 'site_name'])['q'].agg(['mean', 'median', 'min', 'max', lambda r: np.percentile(r, 25), lambda r: np.percentile(r, 75)]) \
    .rename(columns={stat: f'q_{stat}' for stat in ['mean', 'median', 'min', 'max', '<lambda_0>', '<lambda_1>']}).reset_index()

q_stats = q_stats.rename(columns={'q_<lambda_0>': 'q_25', 'q_<lambda_1>': 'q_75'})

for file in glob(os.path.join(velocities_dir, '*.csv')):
    df = pd.read_csv(file, index_col=0, dtype={'site_no': str})
    df = df.rename(columns={df.columns[4]: 'v'})[['datetime', 'site_no', 'site_name', 'v']]
    df['v'] = pd.to_numeric(df['v'], errors='coerce')
    if df['v'].isna().any():
        if df['v'].isna().all():
            print(f'File {file} contained improper values')
            continue
        else:
            df.loc[df['v'].isna(), 'v'] = 0
    if (df['v'] < 0).any().any():
        df.loc[df['v'] < 0, 'v'] = 0.
    if velocities_df.empty:
        velocities_df = df
    else:
        velocities_df = pd.concat([velocities_df, df], axis=0)

# Make a summary table for velocities and combine with discharge stats
v_stats = velocities_df.groupby(['site_no', 'site_name'])['v'].agg(['mean', 'median', 'min', 'max', lambda r: np.percentile(r, 25), lambda r: np.percentile(r, 75)]) \
    .rename(columns={stat: f'v_{stat}' for stat in ['mean', 'median', 'min', 'max', '<lambda_0>', '<lambda_1>']}).reset_index()

v_stats = v_stats.rename(columns={'v_<lambda_0>': 'v_25', 'v_<lambda_1>': 'v_75'})

stats = q_stats.merge(v_stats, how='inner', on=['site_no', 'site_name'])
q_v = discharges_df.merge(velocities_df, how='inner', on=['site_no', 'site_name', 'datetime'])
del discharges_df, velocities_df

# Remove Outliers
# q_v = remove_outliers(q_v, col='v', outlier_threshold=5)
q_v_rel = {'site_no': [], 'm': [], 'b': []}

for site_no, group in q_v.groupby('site_no'):
    # Extract q and v values for the current site_no
    q_values = group['q']
    v_values = group['v']

    # Perform linear regression
    try:
        slope, intercept, _, _, _ = st.linregress(v_values, q_values)
    except:
        continue
    # Store results in dictionary
    q_v_rel['site_no'].append(site_no)
    q_v_rel['m'].append(slope)
    q_v_rel['b'].append(intercept)

q_v_rel = pd.DataFrame(q_v_rel)
# read_nhd = True
# Match points to TDXHydro Network instead of NHD to obtain hydrography attributes
if not read_nhd:
    if not os.path.exists('assigned_usgs_gauges.csv'):
        if not os.path.exists('usgs_vel_site_locations.csv'):
            # Get site information for each site and extract lat/lon data
            site_info_url_pattern = f'https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&format=rdb&site_no='

            site_locations_df = pd.DataFrame()
            for site_no in stats['site_no'].to_list():
                url = site_info_url_pattern + str(site_no)
                r = requests.get(url)
                if r.status_code != 200:
                    print('Get sites failed')
                    continue
                text = r.text
                if 'No sites were found that meet the following criteria...' in text:
                    print("That site code didn't work")
                    continue
                site_info_df = df_from_tab_separated(text)
                lat = string_to_lat_lon(site_info_df['lat_va'].values[0])
                lon = string_to_lat_lon(site_info_df['long_va'].values[0])
                if site_locations_df.empty:
                    site_locations_df = pd.DataFrame({'site_no': site_no, 'lat_gauge': lat, 'lon_gauge': lon}, index=[0])
                else:
                    site_locations_df = pd.concat([site_locations_df,
                                                   pd.DataFrame({'site_no': site_no, 'lat_gauge': lat, 'lon_gauge': lon}, index=[0])
                                                   ]).reset_index(drop=True)
            site_locations_df['lon_gauge'] = -site_locations_df['lon_gauge']
            site_locations_df.to_csv('usgs_vel_site_locations.csv', index=False)
        else:
            site_locations_df = pd.read_csv('usgs_vel_site_locations.csv', dtype={'site_no': str})
        # Match gauge points to stream network
        if not os.path.exists('matched_velocity_gauges.csv'):
            # Get all streams together rather than running workflow on each vpu separately.
            # More memory-intensive but faster
            north_america_df = pd.DataFrame()
            for vpu in glob('tdxhydro/vpu_700/vpu_*_streams.gpkg'):
                print(f'Reading {vpu}')
                if north_america_df.empty:
                    north_america_df = gpd.read_file(vpu)
                else:
                    north_america_df = pd.concat([north_america_df, gpd.read_file(vpu)], axis=0).reset_index(drop=True)

            north_america_df = gpd.GeoDataFrame(north_america_df, geometry='geometry')
            # new_gauges = pd.read_csv('matched_velocity_gauges.csv')
            new_gauges = match_rivers_to_gauges(north_america_df, site_locations_df)
            new_gauges.to_csv('matched_velocity_gauges.csv', index=False)

            # Add slope and upstream area to gauge attributes if matching has just been done otherwise it goes straight
            # Into the numpy array
            new_gauges['nga_id'] = new_gauges['nga_id'].astype(int)
            new_gauges['site_no'] = new_gauges['site_no'].astype(int)
            stats['site_no'] = stats['site_no'].astype(int)
            north_america_df['TDXHydroLinkNo'] = north_america_df['TDXHydroLinkNo'].astype(int)
            new_gauges = new_gauges.merge(north_america_df[['TDXHydroLinkNo', 'Slope', 'USContArea']],
                                          left_on='nga_id', right_on='TDXHydroLinkNo', how='inner')\
                .drop('TDXHydroLinkNo', axis=1)
            new_gauges = new_gauges.merge(stats[['site_no', 'v_mean']], how='inner', on='site_no')
            data_arr = new_gauges[['USContArea', 'Slope', 'v_mean']].values
            new_gauges.to_csv('matched_velocity_gauges.csv', index=False)
        else:
            new_gauges = pd.read_csv('matched_velocity_gauges.csv')
            data_arr = new_gauges[['USContArea', 'Slope', 'v_mean']].values

    else:
        new_gauges = pd.read_csv('assigned_usgs_gauges.csv')
        data_arr = new_gauges[['USContArea', 'Slope', 'v_mean']].values
else:
    nhd_dir = 'NHD_VAA'
    slp_dir = 'NHD_slp'
    vaa_df = pd.DataFrame()
    vaa_comids = [0]
    make_df = False
    if not os.path.exists(nhd_dir):
        os.makedirs(nhd_dir)

    if not os.path.exists(slp_dir):
        os.makedirs(slp_dir)

    if len(os.listdir(slp_dir)) == 0:
        elev_slps = glob(os.path.join('/Users/joshogden/Documents/NHDPlusData', 'NHDPlus*', 'NHDPlus*', 'NHDPlusAttributes',
                                      'elevslope.dbf'))
        for file in elev_slps:
            region = file.split('/')[5]
            df = Dbf5(file).to_dataframe()
            df.to_csv(os.path.join(slp_dir, f'{region}.csv'))

    if len(os.listdir(nhd_dir)) == 0:
        vaa_attrs = glob(os.path.join('/Users/joshogden/Documents/NHDPlusData', 'NHDPlus*', 'NHDPlus*', 'NHDPlusAttributes',
                                      'PlusFlowlineVAA.dbf'))
        for file in vaa_attrs:
            region = file.split('/')[5]
            df = Dbf5(file).to_dataframe()
            df.to_csv(os.path.join(nhd_dir, f'{region}.csv'))

    vaa_comid = 'ComID'
    slp_comid = 'COMID'
    q_comid = 'site_no'
    # vaas = ['TotDASqKM', 'StreamOrde']
    vaas = ['TotDASqKM']
    slp_attrs = ['SLOPE']

    qv_attrs = ['v_mean']
    data_arr = None
    predict_df = pd.DataFrame()
    count = 0
    slp_comids = []

    for vaa, slp in zip(glob(os.path.join(nhd_dir, '*.csv')), glob(os.path.join(slp_dir, '*.csv'))):
        vaa_df = pd.read_csv(vaa, dtype={vaa_comid: str})[[vaa_comid] + vaas]
        vaa_df[vaa_comid] = vaa_df[vaa_comid].apply(lambda x: x.split('.')[0])

        slp_df = pd.read_csv(slp, dtype={slp_comid: str, 'ComID': str})

        if not slp_df.columns.isin([[slp_comid] + slp_attrs]).all():
            slp_df = slp_df.rename(columns={col: col.upper() for col in slp_df.columns})
        slp_df = slp_df[[slp_comid] + slp_attrs]
        slp_df[slp_comid] = slp_df[slp_comid].apply(lambda x: x.split('.')[0])
        slp_comids += slp_df[slp_comid].to_list()
        # slp_df = remove_outliers(slp_df)

        # df = vaa_df.merge(slp_df, how='inner', left_on=vaa_comid, right_on=slp_comid)[[slp_comid] + vaas + slp_attrs]

        # df = df.merge(stats[[q_comid] + qv_attrs], how='inner', left_on=slp_comid, right_on=q_comid) \
        #     .drop([slp_comid, q_comid], axis=1)
        df = slp_df.merge(stats[[q_comid] + qv_attrs], how='inner', left_on=slp_comid, right_on=q_comid) \
            .drop([slp_comid, q_comid], axis=1)
        count += len(df)
        if predict_df.empty:
            predict_df = df
        else:
            predict_df = pd.concat([predict_df, df], axis=0)
        if data_arr is None:
            data_arr = df.to_numpy()
        else:
            data_arr = np.concatenate((data_arr, df.to_numpy()), axis=0)

# --------- Regression -----------
x_attrs = ['TotDASqKM', 'SLOPE']
y_attr = 'v_mean'

# model = linear_model.LinearRegression()
pipe = make_pipeline(linear_model.LinearRegression())
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)


X = data_arr[:, :-2]
y = data_arr[:, -1]
# X = X ** 1/3

# Sqrt slope, log velocity
X[:, 0] = np.log(X[:, 0])
# X[:, 1] = np.sqrt(X[:, 1])
y = np.where(y == 0, 0, np.log(y))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# pipe.fit(X_train, y_train)
# model.fit(X_train, y_train)
# rf_model.fit(X_train, y_train)

pipe.fit(X, y)
rf_model.fit(X, y)

y_predicted = pipe.predict(X)
# y_pred = model.predict(X_test)
y_forest = rf_model.predict(X)

# Create subplots for the independent variables, lrm
fig, axes = plt.subplots(1, X.shape[1], figsize=(6, 3))
dpi = 400

lrm = pipe.named_steps['linearregression']
print('Parameters:')
print(lrm.coef_, lrm.intercept_)
x_labels = ['Drainage Area (sq km)']
y_label = 'Mean Velocity (cfs)'

for i in range(X.shape[1]):
    x = X[:, i]
    x_range = np.linspace(x.min(), x.max(), 100)
    y_pred = np.power(np.exp(1), lrm.coef_[i] * x_range + lrm.intercept_)
    # Plot the independent variable against the dependent variable
    if X.shape[1] > 1:
        axes.scatter(x, np.exp(y))
        axes[i].set_xlabel(x_labels[i])
        axes[i].set_ylabel(y_label)
        axes[i].plot(x_range, y_pred, color='red')
        # Annotate with the equation of the line
        # equation_text = f'y = {lrm.coef_[i]:.4f} * x + {lrm.intercept_:.4f}'
        # axes[i].annotate(equation_text, xy=(0.5, 0.25), xycoords='axes fraction', fontsize=12, color='red')
    else:
        axes.scatter(x, np.exp(y))
        axes.set_xlabel(x_labels[i])
        axes.set_ylabel(y_label)
        axes.plot(x_range, y_pred, color='red')
        # Annotate with the equation of the line
        # equation_text = f'y = {lrm.coef_[i]:.4f} * x + {lrm.intercept_:.4f}'
        # axes[i].annotate(equation_text, xy=(0.5, 0.25), xycoords='axes fraction', fontsize=12, color='red')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.title('Velocity Compared with Upstream Area')
plt.subplots_adjust(top=0.9)

plt.savefig('vel-da.png', format='png', dpi=dpi, bbox_inches='tight')
plt.show()


# Calculate evaluation metrics
mse = mean_squared_error(y, y_predicted)
mae = mean_absolute_error(y, y_predicted)
rmse = np.sqrt(mse)
r_squared_model = pipe.score(X, y)
results = pd.DataFrame({
    'Mean Squared Error': mse,
    'Mean Absolute Error': mae,
    'Root Mean Squared Error': rmse,
    'R^2': r_squared_model
}, index=[0])
results.to_csv('metrics.csv', index=False)

# r_squared_model = model.score(X_test, y_test)
# mse_forest = mean_squared_error(y, y_forest)
# r2_forest = r2_score(y, y_pred)
# feature_importance = rf_model.feature_importances_


# Create a baseline model that predicts the mean of the target variable
# baseline_predictions = [y_train.mean()] * len(y_test)

# Calculate R-squared for the baseline model
# r_squared_baseline = r2_score(y_test, baseline_predictions)

# Create subplots for the independent variables, random forest
x_labels = ['Ln Drainage Area (sq km)']
y_label = 'Ln Mean Velocity (cfs)'
fig, axes = plt.subplots(1, X.shape[1], figsize=(6, 3))
lrm = pipe.named_steps['linearregression']
for i in range(X.shape[1]):
    x = X[:, i]
    x_range = np.linspace(x.min(), x.max(), 100)
    y_pred = lrm.coef_[i] * x_range + lrm.intercept_
    # Plot the independent variable against the dependent variable
    if X.shape[1] > 1:
        axes[i].set_xlabel(x_labels[i])
        axes[i].set_ylabel(y_label)
        axes[i].plot(x_range, y_pred, color='red')
        # Annotate with the equation of the line
        equation_text = f'y = {lrm.coef_[i]:.4f} * x + {lrm.intercept_:.4f}'
        axes[i].annotate(equation_text, xy=(0.5, 0.25), xycoords='axes fraction', fontsize=12, color='red')
    else:
        axes.scatter(x, y)
        axes.set_xlabel(x_labels[i])
        axes.set_ylabel(y_label)
        axes.plot(x_range, y_pred, color='red')
        # Annotate with the equation of the line
        equation_text = f'y = {lrm.coef_[i]:.4f} * x + {lrm.intercept_:.4f}'
        axes.annotate(equation_text, xy=(0.5, 0.25), xycoords='axes fraction', fontsize=12, color='red')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.suptitle('Velocity vs. Drainage Area (log-transformed)')
plt.subplots_adjust(top=0.9)
plt.savefig('vel-da-log.png', format='png', dpi=dpi, bbox_inches='tight')
plt.show()


# Create the figure and subplots
fig, axes = plt.subplots(2, X.shape[1], figsize=(6 * X.shape[1], 6))  # 2 rows, 1 column

for i in range(X.shape[1]):
    x = X[:, i]
    x_range = np.linspace(x.min(), x.max(), 100)
    y_pred = np.power(np.exp(1), lrm.coef_[i] * x_range + lrm.intercept_)
    y_pred_log = lrm.coef_[i] * x_range + lrm.intercept_
    equation_text = f'y = {lrm.coef_[i]:.4f}x \u2013 {abs(lrm.intercept_):.4f}'
    if X.shape[1] > 1:
        # Linear Regression Plot
        axes[0, i].scatter(x, np.exp(y), color='blue')
        axes[0, i].plot(x_range, y_pred, color='red')
        # axes[0, i].set_xlabel(x_labels[i])
        axes[0, i].set_xlabel(x_labels[i].replace('Ln ', ''))
        axes[0, i].set_ylabel(y_label.replace('Ln ', ''))
        axes[0, i].set_title('Velocity vs. Drainage Area')

        # Log-Linear Regression Plot
        axes[1, i].scatter(x, y, color='green')  # Use np.log for the dependent variable
        axes[1, i].plot(x_range, y_pred_log, color='red')
        axes[1, i].set_xlabel(x_labels[i])
        axes[1, i].set_ylabel(y_label)
        axes[1, i].set_title('Velocity vs. Drainage Area (natural-log-transformed)')
        axes[1, i].annotate(equation_text, xy=(0.5, 0.25), xycoords='axes fraction', fontsize=12, color='red')
    else:
        # Linear Regression Plot
        axes[0].scatter(x, np.exp(y), color='blue')
        axes[0].plot(x_range, y_pred, color='red')
        # axes[0, i].set_xlabel(x_labels[i])
        axes[0].set_xlabel(x_labels[i].replace('Ln ', ''))
        axes[0].set_ylabel(y_label.replace('Ln ', ''))
        axes[0].set_title('Velocity vs. Drainage Area')

        # Log-Linear Regression Plot
        axes[1].scatter(x, y, color='green')  # Use np.log for the dependent variable
        axes[1].plot(x_range, y_pred_log, color='red')
        axes[1].set_xlabel(x_labels[i])
        axes[1].set_ylabel(y_label)
        axes[1].set_title('Velocity vs. Drainage Area (natural-log-transformed)')
        axes[1].annotate(equation_text, xy=(0.5, 0.25), xycoords='axes fraction', fontsize=12, color='red')

# Adjust spacing between subplots
plt.tight_layout()

# Save or show the plot
plt.savefig('linear_vs_loglinear.png', format='png', dpi=dpi, bbox_inches='tight')
plt.show()


# Scale attributes
# predict_scaled = predict_df[vaas + slp_attrs].to_numpy()
# predict_scaled_means = np.mean(predict_scaled, axis=0)
# # for i in range(predict_scaled.shape[1] - 1):
# #     predict_scaled[:, i] = predict_scaled[:, i] / predict_scaled_means[i]
# predict_predictions = pipe.predict(predict_scaled)
# plot_arr = predict_df[vaas + slp_attrs].to_numpy()
# plot_arr = np.column_stack((plot_arr, predict_predictions))
# # plot_arr = remove_outliers(plot_arr, col=0)
# # plot_arr = remove_outliers(plot_arr, col=1)
# fig, axes = plt.subplots(1, predict_scaled.shape[1], figsize=(12, 4))
# lrm = pipe.named_steps['linearregression']
# for i in range(predict_scaled.shape[1]):
#     x = plot_arr[:, i]
#     x_range = np.linspace(x.min(), x.max(), 100)
#     y = plot_arr[:, -1]
#     # Plot the independent variable against the dependent variable
#     if predict_scaled.shape[1] > 1:
#         axes[i].scatter(x, y)
#         axes[i].set_xlabel(x_attrs[i])
#         axes[i].set_ylabel(y_attr)
#     else:
#         axes.scatter(x, y)
#         axes.set_xlabel(x_attrs[i])
#         axes.set_ylabel(y_attr)
#
# # Adjust the spacing between subplots
# plt.tight_layout()
#
# # Show the plot
# plt.show()


# print('Mean Squared Error (MSE):', mse)
# print('Mean Absolute Error (MAE):', mae)
# print('Root Mean Squared Error (RMSE):', rmse)
# print('R-squared (Model):', r_squared_model)
# # print('R-squared (Baseline):', r_squared_baseline)
# print('R-squared (Forest):', r2_forest)
# print('Coefficients:', lrm.coef_)
# print('Forest Importances:', feature_importance)
# todo: look at scalars, data engineering, normalization, min at 0 max at 1,
#       everything else at ratio, scikitlearn preprocessing
