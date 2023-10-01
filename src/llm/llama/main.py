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
try:
    from src.baseline.baseline_functions import *
except Exception:
    pass
try:
    from rich.progress import track
except Exception:
    pass
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
import gc
import os
from tqdm import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
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
    data_processed = []
    for i,d in track(enumerate(data),total=len(data)):
        result = process(d,add_instructions=add_instructions)
        if result is not None:
            data_processed.append(result)
    # with multiprocessing.Pool() as p:
    #     data_processed: List[dict] = [e for e in p.map(partial(process,add_instructions=add_instructions), data) if e is not None]
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

def get_tokens(data, tokenizer):
    Ltokens = []
    for i, d in tqdm(enumerate(data)):
        gc.collect()
        torch.cuda.empty_cache()
        text = d['text']
        n_tokens = len(tokenizer.tokenize(text))
        Ltokens.append(n_tokens)
    return Ltokens

def get_max_mix(token_lengths, tokenizer, pipeline):
    min_token_length = min(token_lengths)
    max_token_length = max(token_lengths)
    
    max_work = 0
    min_not_work = float('inf')
    
    while min_token_length < max_token_length:
        gc.collect()
        torch.cuda.empty_cache()
        mid_token_length = (min_token_length + max_token_length) // 2
        text = "hello " * mid_token_length  # Create a test text of the desired length
        try:
            [answer] = pipeline(
                text,
                do_sample=True,
                top_k=1,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=1024,
            )
            del answer
            # If the code above works, update max_work and adjust the search range
            max_work = mid_token_length
            min_token_length = mid_token_length + 1
        except Exception as e:
            # If the code above raises an exception, update min_not_work and adjust the search range
            min_not_work = mid_token_length
            max_token_length = mid_token_length - 1
    return (max_work, min_not_work)

def get_max_tokens(path_descriptions: Path, model_name: str = "meta-llama/Llama-2-13b-chat-hf", token: str = "", start: int = 0, end: int = -1):
    print('Start')
    if token != "":
        login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    double_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=double_quant_config
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    with open(path_descriptions) as f:
        data = json.load(f)
    if end == -1:
        end = len(data)
    data = data[start:end]
    
    token_lengths = get_tokens(data, tokenizer)
    (max_work, min_not_work) = get_max_mix(token_lengths, tokenizer, pipeline)
    with open(path_descriptions.parent / f"max_tokens_v100l_chunk_{start}.json",'w') as f:
        json.dump({
            "min_not_work": min_not_work,
            "max_work": max_work,
            "number_of_tokens": token_lengths
        },f)
        

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
    # preprocess_data("eclipse_clear.json", Path("data"),few_shots=False,id="")
    # preprocess_data("eclipse_clear.json", Path("data"),few_shots=True,id="_few_shots")
    path_data = Path("/project/def-aloise/rmoine/llm_data.json")
    # model="TheBloke/Llama-2-13B-GPTQ"
    get_max_tokens(path_data,token="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",start=0,end=24225)
    get_max_tokens(path_data,token="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",start=24225,end=48450)
    get_max_tokens(path_data,token="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",start=48450,end=72676)
    # compute_metrics(path_data.parent / "predictions.json")
