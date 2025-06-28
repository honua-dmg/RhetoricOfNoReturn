import os
import pandas as pd

# --- CONFIGURATION ---
FEATURES_DIR = "features"
# Make sure this points to your final, pruned events file
EVENTS_FILE = "events.csv" 
FINAL_OUTPUT_FILE = "final_model_data.csv"

def aggregate_weekly_features():
    """
    Reads all feature files, aggregates them into weekly bins for each event,
    and saves a single, model-ready CSV file.
    """
    if not os.path.exists(FEATURES_DIR):
        print(f"Error: Directory '{FEATURES_DIR}' not found. Please run feature_engineering.py first.")
        return

    try:
        events_df = pd.read_csv(EVENTS_FILE)
    except FileNotFoundError:
        print(f"Error: Events file '{EVENTS_FILE}' not found. Please double-check the filename.")
        return

    all_event_data = []
    feature_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('.csv')]
    print("Starting weekly aggregation...")

    for filename in feature_files:
        event_id = filename.replace('.csv', '')
        print(f"  -> Aggregating {event_id}...")
        
        filepath = os.path.join(FEATURES_DIR, filename)
        df = pd.read_csv(filepath)

        # Convert date column to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        
        df['date'] = df['date'].dt.tz_localize(None) # <-- ADD THIS LINE

        # Get the event date from our main events file
        event_info = events_df[events_df['Event_ID'] == event_id]
        if event_info.empty:
            print(f"     Warning: No event info found for {event_id} in {EVENTS_FILE}. Skipping.")
            continue
        
        event_date = pd.to_datetime(event_info['Event_Date'].iloc[0])

        # --- Calculate "Week Before Event" ---
        df['days_before_event'] = (event_date - df['date']).dt.days
        df['week_before_event'] = -(df['days_before_event'] // 7)

        # Filter to only include the 12-week lead-up
        df = df[(df['week_before_event'] >= -12) & (df['week_before_event'] <= -1)]
        
        if df.empty:
            print(f"     Warning: No articles found within the 12-week lead-up for {event_id}.")
            continue

        aggregations = {
            'sentiment_compound': ['mean', 'max', 'min', 'std'],
            'rhetoric_score': ['mean', 'max', 'sum'],
            'certainty_score': ['mean', 'sum'],
            'headline': ['count'] 
        }

        weekly_df = df.groupby('week_before_event').agg(aggregations).reset_index()
        
        weekly_df.columns = ['_'.join(col).strip() for col in weekly_df.columns.values]
        weekly_df = weekly_df.rename(columns={'week_before_event_': 'week_before_event', 'headline_count': 'article_count'})

        weekly_df['Event_ID'] = event_id
        weekly_df['Outcome'] = event_info['Outcome'].iloc[0]
        
        all_event_data.append(weekly_df)

    if not all_event_data:
        print("\nError: No data was aggregated. Please check your input files and paths.")
        return

    final_df = pd.concat(all_event_data, ignore_index=True)

    id_cols = ['Event_ID', 'Outcome', 'week_before_event', 'article_count']
    feature_cols = sorted([col for col in final_df.columns if col not in id_cols])
    final_df = final_df[id_cols + feature_cols]

    final_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8')

    print("\n--- Aggregation Complete ---")
    print(f"Final model-ready data saved to '{FINAL_OUTPUT_FILE}'")
    print(f"The final dataset has {len(final_df)} rows.")

if __name__ == "__main__":
    aggregate_weekly_features()