import os.path
import re
from datetime import datetime

import pandas as pd
import requests


UNIT_LOOKUP = {
    'feet per second': 'fps',
    'cubic feet per second': 'cfs',
    'miles per hour': 'mph'
}

def df_from_tab_separated(txt: str):
    # Get each row of the text, splitting on new lines
    txt_rows = txt.strip().split('\n')

    # Loop through rows until the first row that doesn't begin with # is reached, saving the row number, which is the
    # beginning of the table
    row = '#'
    data_begin = 1
    while row[0] == '#':
        row = txt_rows[data_begin - 1]
        data_begin += 1

    # Get all the rows after that table beginning row and join them into data, then get the first row as the column names
    # and everything after the third row as the data. Finally, combine those into a dataframe
    data = '\n'.join(txt.split('\n')[data_begin - 2:])
    header = data.strip().split('\n')[0].replace('\r', '')
    rows = data.strip().split('\n')[2:]
    column_names = header.split('\t')
    data_rows = [row.replace('\r', '').split('\t') for row in rows]
    if len(data_rows[0]) < len(column_names):
        data_rows = [d + ['' for i in range(len(column_names) - len(data_rows[0]))] for d in data_rows]
    df = pd.DataFrame(data_rows, columns=column_names)
    return df


def get_values_from_usgs(site_type_codes: list,
                         parameter_codes: list,
                         start_date: datetime,
                         end_date: datetime,
                         sites_url: str,
                         data_url: str,
                         write_dir: str,
                         filter_ids: list = None
                         ):
    date_format = "%Y-%m-%d"
    start_date_str = start_date.strftime(date_format)
    end_date_str = end_date.strftime(date_format)
    period = (end_date - start_date).days
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    if filter_ids:
        formatted_ids = ','.join(filter_ids)
        get_sites_url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={formatted_ids}"
        get_sites = requests.get(get_sites_url)
        if get_sites.status_code != 200:
            print('Get site names failed')
            return
        text = get_sites.text
        available_sites_df = df_from_tab_separated(text)
    else:
        for st_cd in site_type_codes:
            sites_url += f'&site_tp_cd={st_cd}'
        for pm_cd in parameter_codes:
            sites_url += f'&index_pmcode_{pm_cd}=1'

        sites_url += f'&range_selection=days' \
               f'&period={period}' \
               f'&begin_date={start_date_str}' \
               f'&end_date={end_date_str}'
        r = requests.get(sites_url)
        if r.status_code != 200:
            print('Get sites failed')
            return
        text = r.text
        if 'No sites were found that meet the following criteria...' in text:
            print("Those parameter codes didn't work")
            return
        available_sites_df = df_from_tab_separated(text)
    if filter_ids:
        available_sites_df = available_sites_df.loc[available_sites_df['site_no'].isin(filter_ids)]
    for site_no, station_nm in zip(available_sites_df['site_no'], available_sites_df['station_nm']):
        i = 0
        text = "No sites found matching all criteria"
        for param_cd in parameter_codes:
            d_url = data_url + f'?sites={site_no}&parameterCd={param_cd}' \
                               f'&startDT={start_date_str}&endDT={end_date_str}' \
                               f'&siteStatus=all&format=rdb'
            r = requests.get(d_url)
            if r.status_code != 200:
                print('Get data failed')
                return
            text = r.text
            if text.find("No sites found matching all criteria") == -1:
                break
        if text.find("No sites found matching all criteria") != -1:
            print(f"Site {station_nm} couldn't find any data")
            continue
        param_desc = re.search(r'\d+\s+\d+\s+(.*)\n', text.split('#    TS_ID       Parameter Description')[1]).group(1)
        print(param_desc)
        unit_str = ''
        for unit in UNIT_LOOKUP.keys():
            if unit in param_desc:
                unit_str = f'_{UNIT_LOOKUP[unit]}'
        data_df = df_from_tab_separated(text)
        data_df['site_name'] = station_nm
        data_df['param_code'] = param_cd
        data_df['param_desc'] = param_desc
        data_df.to_csv(f'{write_dir}/{site_no}{unit_str}.csv')


if __name__ == "__main__":
    base_url = f'https://waterdata.usgs.gov/nwis/uv/?referred_module=sw&format=rdb_station_file&group_key=NONE' \
               f'&date_format=YYYY-MM-DD&rdb_compression=file' \
               f'&list_of_search_criteria=site_tp_cd%2Crealtime_parameter_selection'

    data_base_url = f'https://waterservices.usgs.gov/nwis/iv/'

    st_tps = ['ST', 'ST-CA']
    vel_pm_cds = ['72190', '00055', '81904', '72321', '72322', '72294', '72254', '72196', '72323', '72255', '72149']

    get_values_from_usgs(st_tps, vel_pm_cds, datetime(2023, 6, 8), datetime(2023, 6, 15), base_url, data_base_url,
                         "Velocities")
