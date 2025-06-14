import pandas as pd

def ssov(ts_topics):
    daily_totals = (
    ts_topics
      .groupby('timestamp')['uses']
      .sum()
      .rename('total_uses')
    )

    ts_topics = ts_topics.merge(daily_totals, on='timestamp')
    ts_topics['sov'] = ts_topics['uses'] / ts_topics['total_uses']
    ts_topics['sov_pct'] = (ts_topics['sov'] * 100).round(1)

    return ts_topics[['timestamp', 'topic', 'sov', 'sov_pct']].sort_values(by='timestamp')