import google.generativeai as genai
import pandas as pd
import json
import time
from tqdm import tqdm
import sys
import re

def main():
    API_KEY = *hidden for privacy*
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1, 
        )
    )

    print("Loading transcripts dataset...")
    df = pd.read_csv('claimify_transcripts_sample.csv')
    transcripts = df[['transcript_video_id', 'transcript_content_punctuated']].to_dict('records')
    
    output_file = 'claimify_extracted_transcripts.csv'

    # Check if we have already processed some videos to resume
    processed_vids = set()
    import os
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            processed_vids = set(existing_df['transcript_video_id'].dropna().unique())
            print(f"Found {len(processed_vids)} already processed videos. Resuming...")
        except Exception:
            pass
    else:
        # Create/overwrite the output file with headers if it doesn't exist
        pd.DataFrame(columns=['transcript_video_id', 'claim']).to_csv(output_file, index=False)
    
    print(f"Loaded {len(transcripts)} massive video transcripts.")
    print("Processing ONE transcript at a time due to text volume.")
    print("Starting API extraction...\n")

    system_instruction = """
    You are the Claimify algorithm. Your task is to read full-length transcripts from YouTube political/commentary videos and extract all verifiable factual claims made by the speaker.
    
    A verifiable factual claim is:
    1. Objective: It makes a statement about the real world that can theoretically be proven True or False using evidence.
    2. Specific: It usually contains identifiable entities like people, organizations, dates, locations, or statistics.
    3. Past/Present: It is about past or present events, not predictions about the future.
    4. Quotes count: If the speaker quotes or attributes a statement to someone else (e.g., "The Mayor said X"), that IS a verifiable factual claim because we can verify if the Mayor actually said it.
    5. Not a joke, question, subjective opinion, philosophical rant, or insult.

    Since this is a massive transcript, take your time and extract every distinct claim you can find. 
    
    Return a JSON array of objects. Since I am only feeding you one transcript, you should return a single object in the array. The object must have:
    - "video_id": The exact ID provided.
    - "claims": A list of strings, where each string is a distinct, verifiable factual claim extracted. If there are no claims, this must be an empty list.
    """

    total_claims_found = 0

    for i, t_data in enumerate(tqdm(transcripts, desc="Processing Transcripts")):
        vid = t_data['transcript_video_id']
        
        # Skip if already processed
        if vid in processed_vids:
            continue
            
        # Hard cap at 300 transcripts to control budget
        if len(processed_vids) >= 300:
            print("\nReached the hard cap of 300 processed transcripts. Stopping automatically to protect budget.")
            break
            
        text = str(t_data['transcript_content_punctuated'])
        
        # If the transcript is somehow empty, skip
        if not text or len(text) < 10:
            continue
            
        prompt = f"{system_instruction}\n\nHere is the transcript for Video ID: {vid}\n\nTRANSCRIPT:\n{text}"
        
        success = False
        wait_time = 5 # Start with a 5-second timeout if we hit a wall
        
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                data = json.loads(response.text)
                
                new_rows = []
                for item in data:
                    claims = item.get('claims', [])
                    for claim in claims:
                        new_rows.append({'transcript_video_id': vid, 'claim': claim})
                        total_claims_found += 1
                
                if new_rows:
                    pd.DataFrame(new_rows).to_csv(output_file, mode='a', header=False, index=False)
                
                success = True
                processed_vids.add(vid)
                break
            except Exception as e:
                err_str = str(e)
                # If it's a quota error or 429, wait longer
                if "429" in err_str or "Quota" in err_str:
                    print(f"\nRate limit hit on video {vid}. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    wait_time *= 2 # Exponential backoff: 15s -> 30s -> 60s
                else:
                    print(f"\nError on video {vid}: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                
        if not success:
            print(f"\nFailed to process video {vid} after 3 attempts. Skipping.")
            
        # Very small pacing since we are on the paid tier now
        time.sleep(0.5)

    print(f"\n✅ Finished processing all {len(transcripts)} transcripts!")
    print(f"Total factual claims extracted: {total_claims_found}")
    print(f"Saved all claims to {output_file}")

if __name__ == '__main__':
    main()
