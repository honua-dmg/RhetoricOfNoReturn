# FINAL feature_engineering.py with Capping
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- CONFIGURATION ---
RAW_DATA_DIR = "raw_data"
FEATURES_DIR = "features"
EVENTS_FILE = "events.csv"
ARTICLE_CAP = 500  # The maximum number of articles to use from any single event
RANDOM_STATE = 42 # For reproducible random sampling

# Keyword definitions remain the same
RHETORICAL_KEYWORD_TIERS = {
    1: ['tensions', 'dispute', 'protest', 'unrest', 'sanctions', 'diplomat', 'condemn', 'concern', 'warns'],
    3: ['threat', 'mobilize', 'mobilization', 'mobilizing', 'drill', 'exercise', 'border', 'violation', 'standoff', 'brink'],
    5: ['ultimatum', 'attack', 'airstrike', 'invasion', 'casualties', 'clashes', 'imminent', 'offensive', 'shelling']
}
CERTAINTY_KEYWORDS = {
    1: ['will', 'confirms', 'is', 'fact', 'order', 'declares', 'announces', 'instructs', 'guarantees', 'unquestionably'],
   -1: ['could', 'may', 'might', 'suggests', 'appears', 'reportedly', 'seems', 'potential', 'unconfirmed', 'possibly', 'perhaps']
}

def create_all_features():
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)

    events_df = pd.read_csv(EVENTS_FILE)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    print("--- Starting Full Feature Engineering (with Capping) ---")

    for _, event in events_df.iterrows():
        event_id = event['Event_ID']
        filepath = os.path.join(RAW_DATA_DIR, f"{event_id}.csv")

        if not os.path.exists(filepath):
            continue
            
        print(f"  -> Processing {event_id}...")
        df = pd.read_csv(filepath)
        
        # --- NEW CAPPING LOGIC ---
        if len(df) > ARTICLE_CAP:
            print(f"     Event has {len(df)} articles. Capping to {ARTICLE_CAP} via random sample.")
            df = df.sample(n=ARTICLE_CAP, random_state=RANDOM_STATE)
        

        df['text_to_analyze'] = df['headline'].fillna('') + ' ' + df['snippet'].fillna('')

        country_keywords = {kw.lower() for kw in [event['Country_1'], event['Country_2']] if pd.notna(kw)}
        actor_keywords = set()
        if pd.notna(event["Key_Individuals"]):
            actor_keywords.update([kw.strip().lower() for kw in event["Key_Individuals"].split(',')])
        if pd.notna(event["Key_Groups_Entities"]):
            actor_keywords.update([kw.strip().lower() for kw in event["Key_Groups_Entities"].split(',')])

        def process_row(text):
            lower_text = text.lower()
            sentiment = sentiment_analyzer.polarity_scores(text)['compound']
            rhetoric = sum(p for p, kws in RHETORICAL_KEYWORD_TIERS.items() for kw in kws if kw in lower_text)
            certainty = sum(p for p, kws in CERTAINTY_KEYWORDS.items() for kw in kws if kw in lower_text.split())
            relevance = sum(1 for kw in country_keywords if kw in lower_text) + sum(1 for kw in actor_keywords if kw in lower_text)
            return sentiment, rhetoric, certainty, relevance

        df[['sentiment_compound', 'rhetoric_score', 'certainty_score', 'relevance_score']] = df['text_to_analyze'].apply(
            lambda x: pd.Series(process_row(x))
        )
        
        df = df.drop(columns=['text_to_analyze'])
        features_filepath = os.path.join(FEATURES_DIR, f"{event_id}.csv")
        df.to_csv(features_filepath, index=False, encoding='utf-8')

    print("\n--- Feature Engineering Complete ---")

if __name__ == "__main__":
    create_all_features()