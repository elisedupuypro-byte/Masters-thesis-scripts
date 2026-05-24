import sqlite3
import pandas as pd
import os

DB_PATH = r"c:\Users\elise\Desktop\Mémoire\sql-analysis\usalowcomments.sqlite"
OUTPUT_DIR = r"c:\Users\elise\Desktop\Mémoire\tfidf-nmf\csv_sources"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "usa_youtube_comments_full.csv")

def export_for_nmf():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    conn = sqlite3.connect(DB_PATH)
    
    # Query to get text, formatted date, and channel_title for comments
    query = """
    SELECT 
        c.comment_text_display as text,
        strftime('%d-%m-%Y', c.comment_published_at) as date,
        ch.channel_title
    FROM comment c
    JOIN video v ON c.comment_video_id = v.video_id
    JOIN channel ch ON v.video_channel_id = ch.channel_id
    WHERE c.comment_text_display IS NOT NULL
    """
    
    print("Extracting 484k comments from the database...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Clean newlines from text to prevent CSV breaking
    df['text'] = df['text'].str.replace('\n', ' ').str.replace('\r', '')
    
    # Tool uses ';' as default separator (from params.py)
    print(f"Exporting to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8')
    print(f"Successfully exported {len(df)} comments.")

if __name__ == "__main__":
    export_for_nmf()
