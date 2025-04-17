import pandas as pd

# Load the valhalla LLM classified tweets
df = pd.read_csv("data/cleaned_tweets_with_labels_v_b64.csv")

# Define variants for United States
us_variants = {'United States of America', 'United States'}

# Filter DataFrame for rows where 'country' matches one of the variants
us_df = df[df['country'].isin(us_variants)]

# begin ==== This code checks if all rows with 'state' value have a 'state_code' value and vice versa ====
# Filter rows where 'state' is not null
#state_not_null = us_df[us_df['state'].notna()]

# Check if all corresponding 'state_code' values are also not null
#all_have_state_code = state_not_null['state_code'].notna().all()

#print("All rows with a 'state' also have a 'state_code':", all_have_state_code)

# Filter rows where 'state_code' is not null
#state_not_null = us_df[us_df['state_code'].notna()]

# Check if all corresponding 'state' values are also not null
#all_have_state_code = state_not_null['state'].notna().all()

#print("All rows with a 'state_code' also have a 'state':", all_have_state_code)
# end ==============================================

# Set that stores all the unique political labels
political_labels = {'Non-Political', 'Voting Rights', 'Taxes', 'Free Speech', 'Healthcare', 'Energy Policy',
                    'Foreign Policy', 'Criminal Justice', 'Police Reform', 'Inflation', 'Abortion', 'Immigration',
                    'Welfare', 'Climate Change', 'Gun Control', 'Education', 'Unemployment', 'Social Security',
                    'LGBTQ Rights', 'Minimum Wage', 'Student Loans'}



# Group by each political topic and state and store in dictionary of dictionaries
# nested_dfs[political topic][state code]
# rows with no state code are stored with key='unknown'
nested_dfs = {}
for label in political_labels:
    # Filter by political label
    label_df = us_df[us_df['political_label'] == label]
    
    # Make a copy
    label_df = label_df.copy()

    # Group by state_code (replacing NaN with 'unknown')
    label_df['state_key'] = label_df['state_code'].fillna('unknown')

    # Create a sub-dictionary for this label
    nested_dfs[label] = {
        state: group.drop(columns='state_key')
        for state, group in label_df.groupby('state_key')
    }


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    return sia.polarity_scores(text)

nyc_voting_rights_df = nested_dfs['Voting Rights']['NY'].copy()

nyc_voting_rights_df['sentiment'] = nyc_voting_rights_df['clean_tweet'].apply(get_sentiment_scores)


def assign_likert(score):
    compound = score['compound']
    if compound >= 0.7:
        return 'strongly agree'
    elif compound >= 0.4:
        return 'agree'
    elif compound >= 0.1:
        return 'slightly agree'
    elif compound <= -0.7:
        return 'strongly disagree'
    elif compound <= -0.4:
        return 'disagree'
    elif compound <= -0.1:
        return 'slightly disagree'
    else:
        return 'neutral'

nyc_voting_rights_df['likert'] = nyc_voting_rights_df['sentiment'].apply(assign_likert)

nyc_voting_rights_df.to_csv("./data/nyc_voting_rights_likert.csv", index=False)