import pandas as pd

def aggregate_time_series(
    df,
    timestamp_col='timestamp',
    topic_col='topic_id_nmf',
    sentiment_col='sentiment_label',
    hashtag_col='hashtags_list',
    freq='D',
    rolling_window=7):
    """
    Build daily (or hourly) time‑series for:
      1. topic counts + rolling mean + pct growth
      2. sentiment mix
      3. hashtag counts (if hashtag_col exists)

    Returns a dict of DataFrames: {
      'topics': ts_topics,
      'sentiment': ts_sentiment,
      'hashtags': ts_hashtags (if hashtag_col in df)
    }
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Topic counts
    ts_topics = (
        df
        .groupby([pd.Grouper(key=timestamp_col, freq=freq), topic_col])
        .size()
        .reset_index(name='uses')
    )

    # rolling mean and pct growth
    ts_topics['rolling_mean'] = (
        ts_topics
        .groupby(topic_col)['uses']
        .transform(lambda x: x.rolling(rolling_window, 1).mean())
    )

    ts_topics['uses_lag'] = (
        ts_topics
        .groupby(topic_col)['uses']
        .shift(rolling_window)
    )

    ts_topics['pct_growth'] = (
        (ts_topics['uses'] - ts_topics['uses_lag'])
        / ts_topics['uses_lag']
    ).fillna(0)

    #  Sentiment mix
    ts_sentiment = (
        df
        .groupby([pd.Grouper(key=timestamp_col, freq=freq), sentiment_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    result = {
        'topics': ts_topics,
        'sentiment': ts_sentiment
    }

    # Hashtag counts (if present)
    if hashtag_col in df:
        # explode if not already
        if df[hashtag_col].dtype == object:
            df_h = df.explode(hashtag_col).dropna(subset=[hashtag_col])
        else:
            df_h = df
        ts_tags = (
            df_h.groupby([pd.Grouper(key=timestamp_col, freq=freq), hashtag_col]).size().reset_index(name='uses')
        )
        result['hashtags'] = ts_tags

    return result

def aggregate_hashtag_time_series(df_h, timestamp_col='Timestamp', hashtag_col='hashtag', freq='D', window=7, k_sigma=2.0):
    """
    Aggregates daily usage of hashtags and detects spikes based on sigma rule.
    """
    df_h = df_h.copy()
    df_h[timestamp_col] = pd.to_datetime(df_h[timestamp_col])

    # Group by date and hashtag
    df_h['date'] = df_h[timestamp_col].dt.floor(freq)
    ts_hashtags = df_h.groupby(['date', hashtag_col]).size().reset_index(name='uses')
    
    # Rolling statistics for spike detection
    ts_hashtags = ts_hashtags.sort_values(['hashtag', 'date'])
    ts_hashtags['rolling_mean'] = ts_hashtags.groupby(hashtag_col)['uses'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    ts_hashtags['rolling_std']  = ts_hashtags.groupby(hashtag_col)['uses'].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    
    # Spike detection
    ts_hashtags['is_spike'] = ts_hashtags['uses'] > (ts_hashtags['rolling_mean'] + k_sigma * ts_hashtags['rolling_std'])

    return ts_hashtags