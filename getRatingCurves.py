import requests
import pandas as pd
from getVelocities import df_from_tab_separated

# Url for all files
rating_curve_url = f'https://waterdata.usgs.gov/nwisweb/get_ratings?file_type=exsa'

# Get response and parse out the text
r = requests.get(rating_curve_url)
if r.status_code == 200:
    txt = r.text

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
header = data.strip().split('\n')[0]
rows = data.strip().split('\n')[2:]
column_names = header.split('\t')
data_rows = [row.split('\t') for row in rows]
df = pd.DataFrame(data_rows, columns=column_names)
print(df)

# The dataframe has a column containing all urls, follow the link at the top to see how it's structured. Loop through
# each url and use that for a request, then run all the same code to extract out the table.
for url in df['url']:
    r1 = requests.get(url)

    # Get the site number from its place in the url. This may need to be changed depending on how the paths for
    # different data attributes are structured. This is important for unique names.
    site_no = url.split('/')[-1].split('USGS.')[1].split('.exsa')[0]
    print(site_no)
    if r.status_code == 200:
        txt = r1.text

    txt_rows = txt.strip().split('\n')
    row = '#'
    data_begin = 1
    while row[0] == '#':
        row = txt_rows[data_begin - 1]
        data_begin += 1
    data = '\n'.join(txt.split('\n')[data_begin - 2:])
    header = data.strip().split('\n')[0]
    rows = data.strip().split('\n')[2:]
    column_names = header.split('\t')
    data_rows = [row.split('\t') for row in rows]
    df = pd.DataFrame(data_rows, columns=column_names)
    df.rename(columns={'INDEP': 'Gage Height (ft)', 'DEP': 'Discharge (ft^3/s)'}, inplace=True)
    df.to_csv(f'RatingCurves/{site_no}.csv')
