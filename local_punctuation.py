import sqlite3
import time
import sys
from tqdm import tqdm
from transformers.pipelines.token_classification import TokenClassificationPipeline
from deepmultilingualpunctuation import PunctuationModel

def main():
    print("="*50)
    print("LOCAL PUNCTUATION PROCESSOR")
    print("="*50)
    
    # 1. Patch transformers library (fixes version issues)
    original_sanitize = TokenClassificationPipeline._sanitize_parameters
    def patched_sanitize(self, **kwargs):
        if 'grouped_entities' in kwargs:
            kwargs['aggregation_strategy'] = "simple" if kwargs.pop('grouped_entities') else "none"
        return original_sanitize(self, **kwargs)
    TokenClassificationPipeline._sanitize_parameters = patched_sanitize

    # 2. Load Model
    print("\nLoading AI Punctuation Model... (this may take a few seconds)")
    model = PunctuationModel()
    
    # 3. Connect to Local Database
    db_path = 'usalowcomments.sqlite'
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    # 4. Setup Database Column
    try:
        cur.execute("ALTER TABLE transcript ADD COLUMN transcript_content_punctuated TEXT")
        print("Created new column 'transcript_content_punctuated' in database.")
    except sqlite3.OperationalError:
        pass # Column already exists
        
    # 5. Find transcripts left to process
    cur.execute("SELECT transcript_video_id, transcript_content FROM transcript WHERE transcript_content IS NOT NULL AND transcript_content_punctuated IS NULL")
    rows = cur.fetchall()
    
    total_left = len(rows)
    if total_left == 0:
        print("\nAll transcripts have already been punctuated! You are 100% done.")
        conn.close()
        return
        
    print(f"\nFound {total_left} transcripts left to process.")
    print("Starting process... You can safely stop this script at any time (Ctrl+C).")
    print("Progress is saved permanently to your local database every 50 transcripts.\n")
    
    updates = []
    
    try:
        for vid, content in tqdm(rows, desc="Processing Transcripts", unit="video"):
            if not content or len(content.strip()) < 5:
                continue
                
            # Process text
            punctuated = model.restore_punctuation(content)
            updates.append((punctuated, vid))
            
            # Save to database every 50 transcripts
            if len(updates) >= 50:
                cur.executemany("UPDATE transcript SET transcript_content_punctuated = ? WHERE transcript_video_id = ?", updates)
                conn.commit()
                updates = []
                
        # Final save for any remaining
        if updates:
            cur.executemany("UPDATE transcript SET transcript_content_punctuated = ? WHERE transcript_video_id = ?", updates)
            conn.commit()
            
        print("\n✅ Processing 100% Complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process stopped manually by user.")
        # Save whatever we have so far before exiting
        if updates:
            cur.executemany("UPDATE transcript SET transcript_content_punctuated = ? WHERE transcript_video_id = ?", updates)
            conn.commit()
            print(f"Saved {len(updates)} recent transcripts before exiting.")
        print("You can run this script again later to continue exactly where you left off.")
        
    finally:
        conn.close()

if __name__ == '__main__':
    main()
