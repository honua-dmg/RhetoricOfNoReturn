import os
import pandas as pd

# --- CONFIGURATION ---
RAW_DATA_DIR = "raw_data"
PROCESSED_DATA_DIR = "processed_data"
ARTICLE_CAP = 500  # The maximum number of articles we'll use per event
RANDOM_STATE = 42  # Ensures our random sample is the same every time we run it

def balance_data():
    """
    Reads all raw CSVs, applies a cap to balance the dataset,
    and saves the results to a new directory.
    """
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    
    if not raw_files:
        print(f"Error: No CSV files found in the '{RAW_DATA_DIR}' directory.")
        return

    print("Balancing dataset using the 'capping' strategy...")
    total_raw_articles = 0
    total_processed_articles = 0

    for filename in raw_files:
        print(f"  -> Processing {filename}...")
        
        # Read the raw data
        raw_filepath = os.path.join(RAW_DATA_DIR, filename)
        df = pd.read_csv(raw_filepath)
        
        num_articles = len(df)
        total_raw_articles += num_articles

        # Apply the capping logic
        if num_articles > ARTICLE_CAP:
            print(f"     Event has {num_articles} articles. Capping to {ARTICLE_CAP}.")
            # Take a reproducible random sample
            processed_df = df.sample(n=ARTICLE_CAP, random_state=RANDOM_STATE)
        else:
            print(f"     Event has {num_articles} articles. Using all of them.")
            processed_df = df
        
        total_processed_articles += len(processed_df)
        
        # Save the new balanced dataframe
        processed_filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        processed_df.to_csv(processed_filepath, index=False, encoding='utf-8')

    print("\n--- Balancing Complete ---")
    print(f"Total articles before balancing: {total_raw_articles}")
    print(f"Total articles after balancing: {total_processed_articles}")
    print(f"Balanced data saved to the '{PROCESSED_DATA_DIR}' folder.")


balance_data()