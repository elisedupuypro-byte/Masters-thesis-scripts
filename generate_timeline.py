import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from dateutil import parser as date_parser

def main():
    print("Connecting to database...")
    conn = sqlite3.connect('usalowcomments.sqlite')
    
    print("Loading channel data...")
    df_channels = pd.read_sql_query("""
        SELECT channel_id, channel_title, channel_published_at
        FROM channel
    """, conn)
    
    # Convert dates to datetime objects using a robust method for mixed timezones
    def safe_parse_date(d):
        if pd.isna(d) or not d: return pd.NaT
        try:
            return pd.to_datetime(date_parser.parse(str(d)), utc=True)
        except:
            return pd.NaT
            
    df_channels['channel_published_at'] = df_channels['channel_published_at'].apply(safe_parse_date)
    
    print("Loading video and transcript data (this may take a moment)...")
    # We join with transcript to see if transcript_content is not null
    df_videos = pd.read_sql_query("""
        SELECT 
            v.video_channel_id, 
            v.video_published_at, 
            CASE WHEN t.transcript_content IS NOT NULL THEN 1 ELSE 0 END as has_transcript
        FROM video v
        LEFT JOIN transcript t ON v.video_id = t.transcript_video_id
    """, conn)
    
    df_videos['video_published_at'] = pd.to_datetime(df_videos['video_published_at'], utc=True, errors='coerce')
    conn.close()

    print(f"Loaded {len(df_channels)} channels and {len(df_videos)} videos.")
    
    # Merge video data with channel titles to get Y-axis labels
    df = df_videos.merge(df_channels[['channel_id', 'channel_title']], left_on='video_channel_id', right_on='channel_id', how='left')
    
    # Sort channels so the plot looks organized (e.g. by channel creation date)
    df_channels = df_channels.sort_values('channel_published_at', ascending=False)
    channel_order = df_channels['channel_title'].tolist()
    
    # Plotting setup
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    
    # Create a mapping from channel title to Y-coordinate
    y_map = {title: i for i, title in enumerate(channel_order)}
    
    print("Generating plot...")
    
    # Separate data for plotting
    has_transcript = df[df['has_transcript'] == 1]
    no_transcript = df[df['has_transcript'] == 0]

    # Plot videos without transcripts (Red)
    y_vals_no = [y_map[c] for c in no_transcript['channel_title'] if pd.notnull(c)]
    x_vals_no = no_transcript['video_published_at'].dropna()
    ax.scatter(x_vals_no, y_vals_no, c='#ff4757', alpha=0.15, s=15, marker='|', label='No Transcript (~10k)')

    # Plot videos with transcripts (Blue/Cyan)
    y_vals_yes = [y_map[c] for c in has_transcript['channel_title'] if pd.notnull(c)]
    x_vals_yes = has_transcript['video_published_at'].dropna()
    ax.scatter(x_vals_yes, y_vals_yes, c='#1e90ff', alpha=0.15, s=15, marker='|', label='With Transcript (~38k)')

    # Plot channel creation dates (Black Dots)
    for idx, row in df_channels.iterrows():
        title = row['channel_title']
        creation_date = row['channel_published_at']
        if pd.notnull(creation_date) and title in y_map:
            y = y_map[title]
            ax.scatter(creation_date, y, c='black', s=100, marker='o', edgecolors='none', zorder=5)
            # Only add to legend once
            if title == channel_order[0]:
                ax.scatter([], [], c='black', s=100, marker='o', edgecolors='none', label='Channel Created')
            
            # Add subtle horizontal line for visual grouping
            ax.axhline(y=y, color='black', alpha=0.1, linestyle='-', zorder=0)

    # Formatting
    ax.set_yticks(range(len(channel_order)))
    ax.set_yticklabels(channel_order, fontsize=12)
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(fontsize=11)
    
    plt.title('YouTube Channel Activity Timeline & Data Collection Density', fontsize=18, pad=20, color='black')
    plt.xlabel('Timeline (Years)', fontsize=14, labelpad=10)
    
    # Legend
    leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=12, facecolor='white', edgecolor='black')
    for lh in leg.legend_handles: 
        lh.set_alpha(1) # Make legend markers fully opaque
        lh.set_sizes([100]) # Make them larger to be visible

    plt.tight_layout()
    
    output_path = "channel_timeline.png"
    plt.savefig(output_path, facecolor='white', bbox_inches='tight')
    print(f"Visualization saved successfully as {output_path}")

if __name__ == '__main__':
    main()
