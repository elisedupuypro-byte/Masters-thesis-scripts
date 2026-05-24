import sqlite3
import random
import re
from tqdm import tqdm
from transformers.pipelines.token_classification import TokenClassificationPipeline
from deepmultilingualpunctuation import PunctuationModel

def parse_duration(duration_str):
    if not duration_str: return 0
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match: return 0
    h, m, s = match.groups()
    total = 0
    if h: total += int(h) * 3600
    if m: total += int(m) * 60
    if s: total += int(s)
    return total

def main():
    # ==========================================
    SAMPLE_SIZE = 1000 
    MIN_DURATION_MINUTES = 10
    MAX_DURATION_MINUTES = 60
    # ==========================================
    
    print("="*60)
    print(f"LOCAL SAMPLE PUNCTUATION (Target: {SAMPLE_SIZE} videos)")
    print(f"Rules: Videos between {MIN_DURATION_MINUTES} and {MAX_DURATION_MINUTES} minutes long.")
    print("="*60)
    
    original_sanitize = TokenClassificationPipeline._sanitize_parameters
    def patched_sanitize(self, **kwargs):
        if 'grouped_entities' in kwargs:
            kwargs['aggregation_strategy'] = "simple" if kwargs.pop('grouped_entities') else "none"
        return original_sanitize(self, **kwargs)
    TokenClassificationPipeline._sanitize_parameters = patched_sanitize

    print("\nLoading AI Punctuation Model... (this may take a few seconds)")
    model = PunctuationModel()
    
    conn = sqlite3.connect('usalowcomments.sqlite')
    cur = conn.cursor()

    try:
        cur.execute("ALTER TABLE transcript ADD COLUMN transcript_content_punctuated TEXT")
    except sqlite3.OperationalError:
        pass 
        
    print("\nScanning database for eligible videos...")
    query = """
        SELECT t.transcript_video_id, t.transcript_content, v.video_duration 
        FROM transcript t
        JOIN video v ON t.transcript_video_id = v.video_id
        WHERE t.transcript_content IS NOT NULL 
        AND t.transcript_content_punctuated IS NULL
    """
    cur.execute(query)
    rows = cur.fetchall()
    
    # Filter by duration using our custom parser
    eligible_rows = []
    min_sec = MIN_DURATION_MINUTES * 60
    max_sec = MAX_DURATION_MINUTES * 60
    
    for vid, content, duration_str in rows:
        duration_sec = parse_duration(duration_str)
        if min_sec <= duration_sec <= max_sec:
            eligible_rows.append((vid, content))
            
    print(f"Found {len(eligible_rows)} videos that meet the {MIN_DURATION_MINUTES}-{MAX_DURATION_MINUTES} minute rule.")
    
    if len(eligible_rows) == 0:
        print("No more eligible transcripts found to punctuate!")
        conn.close()
        return
        
    if len(eligible_rows) > SAMPLE_SIZE:
        print(f"Randomly selecting {SAMPLE_SIZE} transcripts from the pool...")
        sampled_rows = random.sample(eligible_rows, SAMPLE_SIZE)
    else:
        print(f"Processing all {len(eligible_rows)} remaining transcripts...")
        sampled_rows = eligible_rows
        
    updates = []
    
    try:
        for vid, content in tqdm(sampled_rows, desc="Processing", unit="video"):
            if not content or len(content.strip()) < 5:
                continue
                
            punctuated = model.restore_punctuation(content)
            updates.append((punctuated, vid))
            
            if len(updates) >= 50:
                cur.executemany("UPDATE transcript SET transcript_content_punctuated = ? WHERE transcript_video_id = ?", updates)
                conn.commit()
                updates = []
                
        if updates:
            cur.executemany("UPDATE transcript SET transcript_content_punctuated = ? WHERE transcript_video_id = ?", updates)
            conn.commit()
            
        print(f"\n✅ Successfully added punctuation to {len(sampled_rows)} transcripts!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process stopped manually.")
        if updates:
            cur.executemany("UPDATE transcript SET transcript_content_punctuated = ? WHERE transcript_video_id = ?", updates)
            conn.commit()
            
    finally:
        conn.close()

if __name__ == '__main__':
    main()
