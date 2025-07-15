from dotenv import load_dotenv
from IPython.display import Audio
import kagglehub
import numpy as np
import os
import pandas as pd
import subprocess


load_dotenv()
deam_dir = os.getenv("EMOPIA_DIR")


def prepare_deam():
    path = kagglehub.dataset_download("imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music")
    print("Path to dataset files:", path)

    subprocess.run(["cp", "-r", f"{path}/DEAM_Annotations", deam_dir], check=True)
    subprocess.run(["cp", "-r", f"{path}/DEAM_audio", deam_dir], check=True)

    audio_path = deam_dir + "DEAM_audio/MEMD_audio/2.mp3"
    Audio(audio_path)

    # Prepare data
    data = pd.DataFrame(columns=["songName", "valence", "arousal", "path"])

    # Read valence and arousal csv files
    arousal = pd.read_csv(deam_dir + "DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv")
    valence = pd.read_csv(deam_dir + "DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv")
    static_1_2000 = pd.read_csv(deam_dir + "DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv")
    static_2000_2058 = pd.read_csv(deam_dir + "DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv")

    # Concatenate the two static dataframes
    static_2000_2058.index = static_2000_2058.index + len(static_1_2000)
    static_2000_2058.drop(columns=[' valence_ max_mean', ' valence_max_std',
            ' valence_min_mean', ' valence_min_std', ' arousal_max_mean',
            ' arousal_max_std', ' arousal_min_mean', ' arousal_min_std'], inplace=True)

    static = pd.concat([static_1_2000, static_2000_2058], axis=0)

    # Create a dataframe with the desired data
    data["songName"] = arousal["song_id"].astype(str) + ".mp3"

    # We will utilise both the dynamic and song_level annotations, by usind their mean values
    # We will first find the mean of the dynamic annotations and apply a weighted average with the static annotations
    for index, row in arousal.iterrows():
        i = 0
        sum = 0
        for value in row[1:]:  # Skip the first column "song_id"
            if not np.isnan(value):
                sum += value
                i += 1
            else:
                break
        dynamic_mean = sum / i
        static_mean = static.loc[index][' arousal_mean']
        static_mean = (static_mean - 5) / 4 # Normalize the static mean between [-1, 1]

        # Since the dynamic mean is not calculated for the first 15 seconds, we will apply a weighted average
        data.at[index, "arousal"] = round(((i * dynamic_mean + (i + 15) * static_mean) / (i + 15)) / 2, 2)

    for index, row in valence.iterrows():
        i = 0
        sum = 0
        for value in row[1:]:  # Skip the first column "song_id"
            if not np.isnan(value):
                sum += value
                i += 1
            else:
                break
        dynamic_mean = sum / i
        static_mean = static.loc[index][' valence_mean']
        static_mean = (static_mean - 5) / 4 # Normalize the static mean between [-1, 1]

        # Since the dynamic mean is not calculated for the first 15 seconds, we will apply a weighted average
        data.at[index, "valence"] = round(((i * dynamic_mean + (i + 15) * static_mean) / (i + 15)) / 2, 2)

    # Save the path of every audio file
    data["path"] = deam_dir + "DEAM_audio/MEMD_audio/" + data["songName"]

    data.to_csv(deam_dir + "deam.csv")
    print("Dataset with audio saved to 'deam'")