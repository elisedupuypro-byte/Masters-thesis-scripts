import google.generativeai as genai
import pandas as pd
import json
import time
from tqdm import tqdm
import sys

def main():
    API_KEY = "AIzaSyANiYYwHjjea0qBniOH6sIeclT_O0usKeg"
    genai.configure(api_key=API_KEY)

    # Use the latest model
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1, # Low temperature for analytical consistency
        )
    )

    print("Loading comments dataset...")
    df = pd.read_csv('claimify_comments_sample.csv')
    comments = df[['comment_id', 'comment_text_display']].to_dict('records')
    
    BATCH_SIZE = 50
    output_file = 'claimify_extracted_claims.csv'

    # Create/overwrite the output file with headers
    pd.DataFrame(columns=['comment_id', 'claim']).to_csv(output_file, index=False)
    
    print(f"Loaded {len(comments)} comments.")
    print(f"Batching {BATCH_SIZE} comments per API request. Total requests: {len(comments) // BATCH_SIZE + 1}")
    print("Starting API extraction...\n")

    system_instruction = """
    You are the Claimify algorithm. Your task is to read comments from YouTube videos and extract all verifiable factual claims made in the text.
    
    A verifiable factual claim is:
    1. Objective: It makes a statement about the real world that can theoretically be proven True or False using evidence.
    2. Specific: It usually contains identifiable entities like people, organizations, dates, locations, or statistics.
    3. Past/Present: It is about past or present events, not predictions about the future.
    4. Not a joke, question, subjective opinion, or insult.

    Return a JSON array of objects. Each object must have:
    - "comment_id": The exact ID provided.
    - "claims": A list of strings, where each string is a distinct, verifiable factual claim extracted. If there are no claims, this must be an empty list.
    """

    total_claims_found = 0

    for i in tqdm(range(0, len(comments), BATCH_SIZE), desc="Processing Batches"):
        batch = comments[i:i + BATCH_SIZE]
        
        prompt = f"{system_instruction}\n\nHere are the comments to analyze:\n{json.dumps(batch)}"
        
        success = False
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                data = json.loads(response.text)
                
                new_rows = []
                for item in data:
                    cid = item.get('comment_id')
                    claims = item.get('claims', [])
                    for claim in claims:
                        new_rows.append({'comment_id': cid, 'claim': claim})
                        total_claims_found += 1
                
                if new_rows:
                    pd.DataFrame(new_rows).to_csv(output_file, mode='a', header=False, index=False)
                
                success = True
                break
            except Exception as e:
                print(f"\nError on batch {i}: {e}. Retrying in 5 seconds...")
                time.sleep(5)
                
        if not success:
            print(f"\nFailed to process batch {i} after 3 attempts. Skipping to next batch.")
            
        # Pacing to respect the 15 Requests Per Minute limit of the free tier
        time.sleep(4.5)

    print(f"\n✅ Finished processing all comments!")
    print(f"Total factual claims extracted: {total_claims_found}")
    print(f"Saved all claims to {output_file}")

if __name__ == '__main__':
    main()
