from transformers import AutoTokenizer
import transformers
try:
    import torch
except Exception:
    pass
from huggingface_hub import login
from nltk.tokenize import word_tokenize
from pathlib import Path
import pandas as pd
from src.baseline.baseline_functions import *
import json
import multiprocessing
from typing import *
import re
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from functools import partial
def preprocess_data(file_name: str, data_folder: Path, few_shots: bool = True, id: str = ""):
    """Takes the csv file as input, apply the preprocessings and write the resulting data to files"""
    print("Starting preprocessing")
    # open json file
    with open(data_folder / file_name) as f:
        data: List[str] = f.readlines()
    add_instructions = ""
    if few_shots:
        add_instructions = build_few_shot(data)
    with multiprocessing.Pool() as p:
        data_processed: List[dict] = [e for e in p.map(partial(process,add_instructions=add_instructions), data) if e is not None]
    for i in range(len(data_processed)):
        data_processed[i]['idx'] = i
    data_out = data_processed
    # write to a file
    folder = data_folder / "llm"
    folder.mkdir(exist_ok=True)
    print("saving...")
    ## save everything in a metadata file
    with open(folder / f"llm_data{id}.json", "w") as f:
        json.dump(data_out, f, indent=2)

def classify(answer: str) -> int:
    """Return 0 if not severe, 1 if severe and -1 if unknown"""
    pattern_severe = "[sS][eE][vV][eE][rR][eE]"
    pattern_not_severe = "[nN][oO][tT] *"+pattern_severe
    if re.match(pattern_severe, answer) is not None or ("0" in answer and "1" not in answer):
        return 1
    if re.match(pattern_not_severe, answer) is not None or ("1" in answer and "0" not in answer):
        return 0
    return -1

def main(path_descriptions: Path, model: str = "meta-llama/Llama-2-13b-chat-hf", token: str = ""):
    print('Start')
    if token != "":
        login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        ),
        device_map="auto",
    )
    with open(path_descriptions) as f:
        data = json.load(f)
    L = []
    for d in data:
        [answer] = pipeline(
            d['description'],
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=100, # I'll change here
        )
        answer = answer['generated_text']
        if not isinstance(answer, str):
            raise Exception("Unknown result answer: "+str(answer))
        severity = classify(answer)
        L.append({**d, "answer": answer, "severity_pred": severity})
    with open(path_descriptions.parent / "predictions.json", "w") as f:
        json.dump(L,f)
    return L

def compute_metrics(data_path: Path):
    """Taking the path to output prediction json file, it computes the statistics of the predictions
    
    # Arguments
        - data_path: Path, 
    """
    with open(data_path) as f:
        data = json.load(f)
    pred = []
    true = []
    for d in data:
        pred.append(d['severity_pred'])
        true.append(d['severity'])
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true, pred)

    # Compute accuracy
    accuracy = accuracy_score(true, pred)
    # Compute precision
    precision = precision_score(true, pred)

    # Compute recall
    recall = recall_score(true, pred)

    # Compute F1-score
    f1 = f1_score(true, pred)
    
    with open(data_path.parent / "metrics.json", "w") as f:
        json.dump({
            "date_timestamp": datetime.datetime.now().timestamp(),
            "confusion_matrix": conf_matrix.tolist(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
            },f)
    
if __name__ == "__main__":
    import warnings

    # Ignore DeprecationWarning
    warnings.filterwarnings("ignore")
    # logging.basicConfig(filename='/project/def-aloise/rmoine/log-severity.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger('severity')
    # main("TinyPixel/Llama-2-7B-bf16-sharded")
    preprocess_data("eclipse_clear.json", Path("data"),few_shots=True,id="_few_shots")
    preprocess_data("eclipse_clear.json", Path("data"),few_shots=False,id="")
    # path_data = Path("/project/def-aloise/rmoine/severityPrediction/llmm_full_data.json")
    # main(path_data)
    # compute_metrics(path_data.parent / "predictions.json")
