import pandas as pd

from CalcErrFromObserved import calc_metrics_on_all_gauges

assigned_gauges = pd.read_csv('/Users/joshogden/Downloads/whole_world_gauge_assign.csv')
metrics_df = calc_metrics_on_all_gauges(assigned_gauges)
metrics_df.to_csv('v1v2_hindcast_error.csv')
