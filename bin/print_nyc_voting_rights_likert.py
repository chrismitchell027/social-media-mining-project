import pandas as pd

df = pd.read_csv('data/nyc_voting_rights_likert.csv')

import pandas as pd

df = pd.read_csv('data/nyc_voting_rights_likert.csv')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Columns to exclude
columns_to_exclude = [
    'lat', 'long', 'city', 'user_id', 'country',
    'tweet_id', 'state', 'state_code', 'political_label'
]


print(len(df.drop(columns=columns_to_exclude)))
