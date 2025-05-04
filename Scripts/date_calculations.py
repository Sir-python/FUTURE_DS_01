import pandas as pd

def build_date_dim(df, timestamp_col):
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    start = df[timestamp_col].dt.date.min()
    end   = df[timestamp_col].dt.date.max()

    dates = pd.DataFrame({'Date': pd.date_range(start=start, end=end, freq='D')})

    dates['Year']       = dates['Date'].dt.year
    dates['Quarter']    = dates['Date'].dt.quarter
    dates['Month']      = dates['Date'].dt.month
    dates['MonthName']  = dates['Date'].dt.month_name()
    dates['Day']        = dates['Date'].dt.day
    dates['DayOfWeek']  = dates['Date'].dt.day_name()
    dates['IsWeekend']  = dates['DayOfWeek'].isin(['Saturday','Sunday'])
    dates['WeekOfYear'] = dates['Date'].dt.isocalendar().week
    dates['YearMonth']  = dates['Date'].dt.to_period('M').astype(str)

    return dates