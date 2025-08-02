from dotenv import load_dotenv
from IPython.display import Audio
from midi2audio import FluidSynth
import muspy
import os
import pandas as pd
from tqdm import tqdm


load_dotenv()
emopia_dir = os.getenv("EMOPIA_DIR")


def prepare_emopia():
    emopia = muspy.EMOPIADataset(emopia_dir, download_and_extract=True)
    emopia.convert()

    # understand the structure of the dataset
    track = emopia[0]
    name = track.metadata.source_filename
    annotation = track.annotations[0].annotation
    emo_class = int(annotation["emo_class"])
    print("The filename of the first track is:", name)
    print("The annotation of the first track is:", annotation)

    # print the category of the track according to the class of the emotion
    if emo_class == 1:
        print("The emotion of the track is: Excitement")
    elif emo_class == 2:
        print("The emotion of the track is: Anger")
    elif emo_class == 3:
        print("The emotion of the track is: Sadness")
    else:
        print("The emotion of the track is: Calmness")


    # Function to convert MIDI to WAV
    def midi_to_wav(midi_path, wav_path, soundfont_path):
        """
        Converts a MIDI file to WAV format using FluidSynth.
        """
        fs = FluidSynth(soundfont_path)
        fs.midi_to_audio(midi_path, wav_path)

    # Set up input and output directories
    midi_dir = emopia_dir + "EMOPIA_2.2/midis"
    wav_dir = emopia_dir + "EMOPIA_2.2/wav"
    soundfont_path = "resources/GeneralUser-GS.sf2"

    # Create output directory if it doesn't exist
    os.makedirs(wav_dir, exist_ok=True)

    # Process each MIDI file in the input directory
    for midi_file in tqdm(os.listdir(midi_dir)):
        if midi_file.endswith(".mid") or midi_file.endswith(".midi"):
            midi_path = os.path.join(midi_dir, midi_file)
            wav_filename = os.path.splitext(midi_file)[0] + ".wav"
            wav_path = os.path.join(wav_dir, wav_filename)

            # Convert MIDI to WAV
            midi_to_wav(midi_path, wav_path, soundfont_path)

    print("All MIDI files have been converted to WAV!")

    # Check that the transformation was successful
    audio_path = emopia_dir + "EMOPIA_2.2/wav/Q1__8v0MFBZoco_0.wav"
    Audio(audio_path)

    # Prepare data
    data = {
        "songName": [],
        "emo_class": [],
        "path": []
    }

    # Base path to your audio files
    base_path = emopia_dir + "EMOPIA_2.2/wav"

    for track in emopia:
        name = (track.metadata.source_filename).replace(".mid", ".wav")
        annotation = track.annotations[0].annotation
        emo_class_num = int(annotation["emo_class"])
        path = os.path.join(base_path, name)

        # Determine the emotion class category
        if emo_class_num == 1:
            emo_class = "Excitement"
        elif emo_class_num == 2:
            emo_class = "Anger"
        elif emo_class_num == 3:
            emo_class = "Sadness"
        else:
            emo_class = "Calmness"

        # Append to data dictionary
        data["songName"].append(name)
        data["emo_class"].append(emo_class)
        data["path"].append(path)

    data = pd.DataFrame(data)
    data.to_csv(emopia_dir + "emopia.csv")
    print("Dataset with audio saved to 'emopia'")