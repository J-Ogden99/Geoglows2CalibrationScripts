import os
from datetime import datetime

import pandas as pd
from sklearn import linear_model

from getVelocities import get_values_from_usgs

pd.set_option('display.max_columns', 20)

base_url = f'https://waterdata.usgs.gov/nwis/uv/?referred_module=sw&format=rdb_station_file&group_key=NONE' \
           f'&date_format=YYYY-MM-DD&rdb_compression=file' \
           f'&list_of_search_criteria=site_tp_cd%2Crealtime_parameter_selection'

data_base_url = f'https://waterservices.usgs.gov/nwis/iv/'

velocities_dir = 'Velocities'
discharges_dir = 'Discharges'
stream_widths_dir = 'StreamWidths'

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

# Todo run function with all english unit ids filtered separately from metric. Get all discharges, maybe stream widths
# Todo too, then use those to run a predictive model for velocity based on Q/stream width (maybe), then get K using the
# Todo conversion methods on the Army Corps site.
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
for file in [os.path.join(discharges_dir, f) for f in os.listdir(discharges_dir)]:
    df = pd.read_csv(file, index_col=0)
    df = df.rename(columns={df.columns[4]: 'q'})[['site_no', 'site_name', 'q']]
    df['q'] = pd.to_numeric(df['q'], errors='coerce')
    if not df['q'].notna().all():
        print(f'File {file} contained improper values')
        continue
    if discharges_df.empty:
        discharges_df = df
    else:
        discharges_df = pd.concat([discharges_df, df], axis=0)
discharge_stats = discharges_df.groupby(['site_no', 'site_name'])['q'].agg(['mean', 'median', 'min', 'max']) \
    .rename(columns={stat: f'q_{stat}' for stat in ['mean', 'median', 'min', 'max']}).reset_index()

reg = linear_model.LinearRegression()
