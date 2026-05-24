import sqlite3
import pandas as pd
import os

DB_PATH = r"c:\Users\elise\Desktop\Mémoire\sql-analysis\usalowcomments.sqlite"
OUTPUT_DIR = r"c:\Users\elise\Desktop\Mémoire\tfidf-nmf\csv_sources"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "usa_youtube_transcripts.csv")

def export_for_nmf(min_length=150):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    conn = sqlite3.connect(DB_PATH)
    
    # Query to get text, formatted date, and channel_title for transcripts
    query = f"""
    SELECT 
        t.transcript_content as text,
        strftime('%d-%m-%Y', v.video_published_at) as date,
        ch.channel_title
    FROM transcript t
    JOIN video v ON t.transcript_video_id = v.video_id
    JOIN channel ch ON v.video_channel_id = ch.channel_id
    WHERE t.transcript_content IS NOT NULL AND length(t.transcript_content) >= {min_length}
    """
    
    print(f"Exporting transcripts (min length {min_length})...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Tool uses ';' as default separator (from params.py)
    df.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8')
    print(f"Exported {len(df)} transcripts to {OUTPUT_FILE}")

if __name__ == "__main__":
    export_for_nmf()
