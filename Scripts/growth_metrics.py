import pandas as pd

# def detect_spikes(
#     ts_topics, 
#     uses_col='uses', 
#     mean_col='rolling_mean', 
#     std_col='rolling_std', 
#     pct_col='pct_growth', 
#     window=7, 
#     k_sigma=2.0, 
#     growth_thresh=0.75):

#     """
#     Flags a spike whenever uses > rolling_mean + k_sigma * rolling_std.
#     """
#     df = ts_topics.copy().sort_values(['topic_id_lda','Timestamp'])
#     df[std_col] = df.groupby('topic_id_lda')[uses_col].transform(lambda x: x.rolling(window,1).std().fillna(0))
#     df['spike_sigma'] = df[uses_col] > (df[mean_col] + k_sigma * df[std_col])
#     df['spike_growth'] = df[pct_col] > growth_thresh
#     df['is_spike'] = df['spike_sigma']

#     return df

def detect_spikes(
    ts_topics,
    topic_col='topic_id_lda',
    uses_col='uses',
    mean_col='rolling_mean',
    std_col='rolling_std',
    pct_col='pct_growth',
    window=7,
    k_sigma=2.0,
    growth_thresh=0.8):
    
    df = ts_topics.copy().sort_values([topic_col,'Timestamp'])
    
    # compute rolling mean & std
    df[std_col] = (
      df.groupby(topic_col)[uses_col].transform(lambda x: x.rolling(window,1).std().fillna(0))
    )
    
    # compute lag & pct growth if not already present
    df['uses_lag'] = df.groupby(topic_col)[uses_col].shift(window)
    df[pct_col] = ((df[uses_col] - df['uses_lag'])/df['uses_lag'].clip(lower=1)).fillna(0)
    
    # sigma rule and growth rule
    df['spike_sigma']  = df[uses_col] > (df[mean_col] + k_sigma * df[std_col])
    df['spike_growth'] = df[pct_col] > growth_thresh
    
    # final spike = both conditions
    df['is_spike']     = df['spike_sigma'] & df['spike_growth']
    
    return df

def detect_spikes_combined(
    ts,
    topic_col='topic_id_nmf',
    uses_col='uses',
    timestamp_col='timestamp',
    window=7,
    k_sigma=2.0,
    growth_thresh=0.5,
    min_lag_uses=10):

    df = ts.copy()
    df = df.sort_values([topic_col, timestamp_col])
    
    # rolling mean & std
    df['rolling_mean'] = (df.groupby(topic_col)[uses_col].transform(lambda x: x.rolling(window, 1).mean()))

    df['rolling_std'] = (df.groupby(topic_col)[uses_col].transform(lambda x: x.rolling(window, 1).std().fillna(0)))

    # lagged uses
    df['uses_lag'] = (df.groupby(topic_col)[uses_col].shift(window))

    # percent growth (clip lag to avoid huge ratios on tiny lag values)
    df['uses_lag_clipped'] = df['uses_lag'].clip(lower=min_lag_uses)
    df['pct_growth'] = ((df[uses_col] - df['uses_lag_clipped']) / df['uses_lag_clipped']).fillna(0)

    # flags
    df['spike_sigma'] = df[uses_col] > (df['rolling_mean'] + k_sigma * df['rolling_std'])
    df['spike_growth'] = df['pct_growth'] > growth_thresh
    
    df['is_spike'] = df['spike_sigma'] & df['spike_growth']

    return df