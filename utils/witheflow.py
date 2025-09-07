import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import ast


load_dotenv()
witheflow_dir = os.getenv("WITHEFLOW_DIR")

def prepare_witheflow():
    # Load the CSV file
    csv_path = os.path.join(witheflow_dir, "emotions.csv")
    df = pd.read_csv(csv_path)
    
    # Make sure 'emotion' column exists
    if 'emotion' not in df.columns:
        raise ValueError("CSV does not contain 'emotion' column")

    va = []
    lb = []
    labels = set()

    for index, row in df.iterrows():
        parts = row['filename'].split("_")
        song_name = "_".join(parts[:-1]) + "_AU_P" + parts[0][1]
        valence = round(np.mean(ast.literal_eval(row['valence'])), 2)
        arousal = round(np.mean(ast.literal_eval(row['arousal'])), 2)
        path = os.path.join(witheflow_dir, "audio", song_name + ".wav")
        for label in str(row['emotion']).split(';'):
            labels.add(label.strip())

        va.append({
            "songName": song_name,
            "valence": valence,
            "arousal": arousal,
            "path": path
        })
        
        lb.append({
            "songName": song_name,
            "labels": row['emotion'],
            "path": path
        })
    
    # Dataframe containing mean valence and arousal for each track
    df_va = pd.DataFrame(va)
    
    # Dataframe containing all emotion labels for each track
    df_lb = pd.DataFrame(lb)

    # Print all unique labels
    print("Unique emotion labels found:")
    for label in sorted(labels):
        print("-", label)
    
    # Save the df
    df_va.to_csv(os.path.join(witheflow_dir, "wtf_va.csv"), index=False)
    print("\nDataFrame with mean valence and arousal saved to wtf_va.csv'")
    df_lb.to_csv(os.path.join(witheflow_dir, "wtf_lb.csv"), index=False)
    print("DataFrame with emotion labels saved to wtf_lb.csv'")

    return labels

prepare_witheflow()