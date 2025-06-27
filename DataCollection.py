import os
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import dotenv
dotenv.load_dotenv()

# --- CONFIGURATION & CONSTANTS ---
LEAD_UP_DAYS = 90
DATA_DIR = "raw_data"
EVENTS_FILE = "events.csv"

# --- API HELPERS ---

def build_query(event):
    """Builds a robust boolean query for the APIs."""
    
    # 1. Country terms (must have at least one)
    country_terms = f'({event["Country_1"]}'
    if pd.notna(event["Country_2"]):
        country_terms += f' OR {event["Country_2"]}'
    country_terms += ')'

    # 2. Key entities (should have at least one)
    actors = []
    if pd.notna(event["Key_Individuals"]):
        actors.extend([f'{name.strip()}' for name in event["Key_Individuals"].split(',')])
    if  pd.notna(event["Key_Groups_Entities"]):
         actors.extend([f'{name.strip()}' for name in event["Key_Groups_Entities"].split(',')])
    
    actor_terms = " AND (" + " OR ".join(actors) + ")" if actors else ""

    # 3. Thematic terms (from the Search_Terms column)
    theme_terms = ""
    if  pd.notna(event["Search_Terms"]):
        themes = [f'{term.strip()}' for term in event["Search_Terms"].split(',')]
        theme_terms = " AND (" + " OR ".join(themes) + ")"

    # Combine all parts
    full_query = country_terms + actor_terms + theme_terms
    return full_query

def fetch_guardian_data(query, start_date, end_date):
    """Fetches data from The Guardian API, handling pagination."""
    print("  -> Fetching from The Guardian...")
    articles = []
    page = 1
    total_pages = 1
    
    while page <= total_pages:
        params = {
            'q': query,
            'from-date': start_date,
            'to-date': end_date,
            'api-key': os.getenv('GUARDIAN_API_KEY'),
            'page-size': 50,
            'page': page,
            'show-fields': 'bodyText'
        }
        try:
            response = requests.get('https://content.guardianapis.com/search', params=params)
            response.raise_for_status()
            data = response.json()['response']
            
            for item in data.get('results', []):
                articles.append({
                    'source': 'The Guardian',
                    'date': item['webPublicationDate'],
                    'headline': item['webTitle'],
                    'snippet': item.get('fields', {}).get('bodyText', '')[:500] # Get first 500 chars of body - why?
                })
            
            total_pages = data.get('pages', 1)
            print(f"     Got page {page}/{total_pages}...")
            page += 1
            time.sleep(1) # IMPORTANT: Respect API rate limits
        except requests.exceptions.RequestException as e:
            print(f"     Error fetching from The Guardian: {e}")
            break
            
    return articles




if __name__ == "__main__":
    """Main function to orchestrate the data acquisition process."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    events_df = pd.read_csv(EVENTS_FILE)

    for _, event in events_df.iterrows():
        event_id = event['Event_ID']
        output_filename = os.path.join(DATA_DIR, f"{event_id}.csv")

        if os.path.exists(output_filename):
            print(f"Skipping {event_id}: Data file already exists.")
            continue

        print(f"\nProcessing Event: {event_id} - {event['Event_Name']}")

        # Prepare dates and query
        event_date = datetime.strptime(event['Event_Date'], '%Y-%m-%d')
        start_datetime = event_date - timedelta(days=LEAD_UP_DAYS)
        
        start_date_str = start_datetime.strftime('%Y-%m-%d')
        end_date_str = event_date.strftime('%Y-%m-%d')
        
        query = build_query(event)
        print(f"  -> Query: {query[:200]}...") # Print first 200 chars of query

        # Fetch data from all sources
        guardian_articles = fetch_guardian_data( query, start_date_str, end_date_str)

        # Combine and save
        all_articles = guardian_articles 
        if all_articles:
            df = pd.DataFrame(all_articles)
            df.to_csv(output_filename, index=False, encoding='utf-8')
            print(f"Success! Saved {len(df)} articles to {output_filename}")
        else:
            print(f"Warning: No articles found for {event_id}. An empty file will not be created.")
