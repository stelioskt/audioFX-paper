from dotenv import load_dotenv
import os
from transformers import Wav2Vec2FeatureExtractor, AutoModel, Wav2Vec2FeatureExtractor, ClapModel, ClapProcessor, AutoProcessor, Qwen2AudioForConditionalGeneration
import torch

from utils.emopia import prepare_emopia
from utils.deam import prepare_deam
from utils.model_eval import model_eval
from utils.extract_results import extract_results
from utils.visualisation import visualize_results


load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
MODEL_DIR = os.getenv("MODEL_DIR")
emopia_dir = os.getenv("EMOPIA_DIR")
deam_dir = os.getenv("DEAM_DIR")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(emopia_dir, exist_ok=True)
os.makedirs(deam_dir, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"Emopia directory: {emopia_dir}")
print(f"DEAM directory: {deam_dir}")

# Define models
mert = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True, output_hidden_states=True)
mert_proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
mert_sr = mert_proc.sampling_rate
mert.to(DEVICE)

clap = ClapModel.from_pretrained("laion/larger_clap_music")
clap_proc = ClapProcessor.from_pretrained("laion/larger_clap_music")
clap_sr = clap_proc.feature_extractor.sampling_rate
clap.to(DEVICE)

qwen = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True, torch_dtype=torch.float16, max_memory={0: "20GiB"}, device_map="auto")
qwen_proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)
qwen_sr = qwen_proc.feature_extractor.sampling_rate
qwen.to(DEVICE)


# Step 1: Dataset Evaluation/Preprocessing
print("Preparing datasets...")
# prepare_emopia()
# prepare_deam()

# Step 2: Model Evaluation
print("Evaluating models...")
# model_eval(mert, mert_proc, mert_sr, clap, clap_proc, clap_sr, qwen, qwen_proc, qwen_sr)

# Step 3: Produce Results
print("Extracting results...")
# extract_results(mert, mert_proc, mert_sr, clap, clap_proc, clap_sr, qwen, qwen_proc, qwen_sr)

# Step 4: Result Analysis
print("Visualising results...")
visualize_results()