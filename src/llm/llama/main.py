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
import numpy as np
import multiprocessing
from typing import *
import re
import matplotlib.pyplot as plt
import matplotlib
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
import math
from tqdm import tqdm
from pretty_confusion_matrix import pp_matrix
from itertools import product
from textwrap import wrap
try:
    import shap
except Exception:
    pass
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def print_args(func):
    def inner(*args,**kwargs):
        print("*"*100)
        print("Start",func.__name__)
        print("With *args",args)
        print("With **kwargs",kwargs)
        print("-"*100)
        return func(*args,**kwargs)
    return inner
def classify(answer: str) -> int:
    """Return 0 if not severe, 1 if severe and -1 if unknown"""
    pattern_severe = "[sS][eE][vV][eE][rR][eE]"
    pattern_not_severe = "[nN][oO][tT] *"+pattern_severe
    if re.match(pattern_not_severe, answer) is not None or ("0" in answer and "1" not in answer):
        return 0
    elif re.match(pattern_severe, answer) is not None or ("1" in answer and "0" not in answer):
        return 1
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

def build_prompt(llama_tokenized_template: List[str], llama_tokenized_description: List[str], template_index_insert: int, tokenizer) -> Tuple[str,List[str]]:
    """We put the template and the description together using the insertion point specified in description_index_insert"""
    ## First make a copy
    tokenized_full_text = llama_tokenized_template[:]
    ## Then remove the <s> token from the description
    description = llama_tokenized_description[1:]
    ## Then insert the description inside the template at the position indicated (after input)
    tokenized_full_text[template_index_insert:template_index_insert] = description
    ## And remove the start token (as it will be put again by the tokenizer)
    tokenized_full_text.pop(0)
    ## Convert back into a sentence
    text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenized_full_text))
    return text,tokenized_full_text

@print_args
def main_inference(path_data_preprocessed: Path, model_name: str = "meta-llama/Llama-2-13b-chat-hf", token: str = "", start: int = 0, end: int = -1, limit_tokens: int = 7364, id_pred: str = ""):
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
    with open(path_data_preprocessed) as f:
        data_preprocessed = json.load(f)
    data = data_preprocessed["data"]
    template = data_preprocessed["template"]
    if end == -1:
        end = len(data)
    data = data[start:end]
    responses = []
    folder_predictions = path_data_preprocessed.parent / "predictions"
    folder_predictions.mkdir(exist_ok=True,parents=True)
    for i, d in tqdm(enumerate(data)):
        gc.collect()
        torch.cuda.empty_cache() #type: ignore
        answer = float('nan')
        severity = float('nan')
        text,tokenized_full_text = build_prompt(
            d["llama_tokenized_template"],
            template["llama_tokenized_description"],
            template["template_index_insert"],
            tokenizer
        )
        n_tokens = len(tokenized_full_text)
        if n_tokens < limit_tokens:
            [answer] = pipeline( #type: ignore
                text,
                do_sample=True,
                top_k=1,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=1024,
                return_full_text=False
            )
            answer = answer['generated_text'] #type: ignore
            if not isinstance(answer, str):
                raise Exception("Unknown result answer: "+str(answer))
            severity = classify(answer)

        responses.append({**d, "answer": answer, f"severity_pred{id_pred}": severity, f"input{id_pred}": text})
        if i%5==0:
            with open(folder_predictions / f"predictions_v100l_chunk_{start}.json", "w") as f:
                json.dump(responses,f)
        
    with open(folder_predictions / f"predictions/predictions_v100l_chunk_{start}.json", "w") as f:
        json.dump(responses,f,indent=2)

def extract_fields_from_json(folder_path: Path, fields: List[str], allow_nan: bool = False, allow_incoherent: bool = False):
    """Aggregates every predictions and true value stored in each json file stored in folder_path with possibility to choose which fields to return
    
    # Arguments
        - folder_path: Path, the path where all of the *.json files are stored. They must contain List[dict] with in dict keys 'binary_severity' and 'severity_pred'
    
    # Returns
        - Dict[str,List[int]], Lists for each field required in fields associated to their field name
    """
    fields_data = {f:[] for f in fields}

    folder_path = Path(folder_path)
    json_file_paths = [file for file in folder_path.glob('*.json')]

    for json_file_path in json_file_paths:
        with open(json_file_path) as f:
            data: List[DataoutDict] = json.load(f)
        for d in data:
            severity_pred_value = d['severity_pred']
            # Check if severity_pred_value is -1 or nan before adding them to the lists
            nan_allow = (not np.isnan(severity_pred_value)) or allow_nan
            incoherent_allow = severity_pred_value != -1 or allow_incoherent
            if nan_allow and incoherent_allow:
                for f in fields:
                    fields_data[f].append(d[f])
    return fields_data

def main_shap(file_examples: Path, folder_out: Path, model_name: str = "meta-llama/Llama-2-13b-chat-hf", token: str = ""):
    if not folder_out.exists():
        folder_out.mkdir(parents=True)
    # open sentences
    with open(file_examples) as f:
        representants = json.load(f)
        mapping = representants['mapping']
        representants = {eval(k):v for k,v in representants['samples'].items()} 
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
    pipeline.config.task_specific_params["text-generation"] = dict(
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
        return_full_text=False
    )
    print("Starting inference")
    for k,Ltext in representants.items():
        for i,text in enumerate(Ltext):
            explainer = shap.Explainer(pipeline, tokenizer)
            shap_values = explainer([text])
            with open(folder_out / f"shap_pred_{k[0]}_true_{k[1]}_sample_{i}.html","w") as f:
                f.write(shap.plots.text(shap_values, display=False))
def compute_metrics(folder_predictions: Path, folder_out: Optional[Path] = None, pred_field: str = "severity_pred", mapping_dict: Optional[dict] = None, limit_tokens: int = 7364):
    """Taking the path of the predictions folder, it computes the statistics with the predictions (confusion matrix, precision, recall, f1-score). The confusion matrix is plotted into a png file
    
    # Arguments
        - folder_predictions: Path, path to the folder where the prediction files are stored
        - folder_out: Path, path to the folder where the statistics will be stored
        - mapping_dict: Optional[Dict], mapping from possible values predicted to name (str or int), default {-2:"Too big query", -1:"Mixed answer", 0: "NON SEVERE", 1: "SEVERE"} and -2 replaces all nan
        
    # Return
        None        
    """
    if folder_out is None:
        folder_out = folder_predictions
    fields_data = extract_fields_from_json(folder_predictions, fields=["bug_id", "binary_severity", pred_field, "description"], allow_nan=True, allow_incoherent=True)
    # Replace Nan by -2
    pred = [-2 if np.isnan(e) else e for e in fields_data[pred_field] ]
    true = [-2 if pred[i] == -2 else fields_data['binary_severity'][i] for i in range(len(fields_data['binary_severity']))]
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true, pred)
    if mapping_dict is None:
        mapping_dict = {-2: f"Too big >={limit_tokens}", -1:"Mixed answer", 0: "NON SEVERE", 1: "SEVERE"}
    # Compute accuracy
    accuracy = accuracy_score(true, pred)
    # Compute precision
    precision: np.ndarray = precision_score(true, pred, average=None) #type: ignore

    # Compute recall
    recall: np.ndarray = recall_score(true, pred, average=None) #type: ignore

    # Compute F1-score
    f1: np.ndarray = f1_score(true, pred, average=None) #type: ignore
    
    with open(folder_out / "metrics.json", "w") as f:
        json.dump({
            "date_timestamp": datetime.datetime.now().timestamp(),
            "confusion_matrix": conf_matrix.tolist(),
            "accuracy": accuracy,
            "precision": precision.tolist(), 
            "recall": recall.tolist(), 
            "f1": f1.tolist() 
            },f)
    possibilities_pred = [-2 if np.isnan(e) else e for e in np.unique(pred)]
    plot_confusion(
        conf_matrix=conf_matrix,
        folder_path=folder_out,
        mapping_dict=mapping_dict,
        unique_values=possibilities_pred,
        limit_tokens=limit_tokens,
        backend="agg" #type: ignore
    )
    # find representants
    find_representant(
        fields_data, 
        folder_out / "representants.json", 
        n_samples=5,
        mapping_dict=mapping_dict,
        limit_tokens=limit_tokens,
        poss=possibilities_pred,
        pred_field=pred_field
    )
    
def plot_confusion(conf_matrix: np.ndarray, folder_path: Optional[Path] = None, mapping_dict: Optional[Dict] = None, unique_values: Optional[List] = None, backend = Optional[Literal['agg']], limit_tokens: int = 7366):
    """Takes the confusion matrix and plots it with totals values (recall is the percentage of the total of each column, precision percentage for the total of each line and accuracy is the percentage at the bottom right)
    Can be used in notebooks just to plot or just to save into a file. See doc of arguments
    
    Original package: https://github.com/wcipriano/pretty-print-confusion-matrix
    Doc of MATLAB inspired confusion matrix plotted here: https://www.mathworks.com/help/deeplearning/ref/plotconfusion.html;jsessionid=7052c44c75f9529f74ccd8187446
    
    # Arguments
        - conf_matrix: np.ndarray, confusion matrix
        - folder_path: Path, path to the folder where the plot will be saved
        - mapping_dict: Optional[Dict], mapping from possible values predicted to name (str or int), default {-2, "Too big query", -1:"Mixed answer", 0: "NON SEVERE", 1: "SEVERE"} and -2 replaces all nan
        - unique_values: Optional[List], list of possible values default [-2, -1, 0, 1]
        - backend: Optional[Literal['agg']], the backend to use, tries to plot on screen (default) if no backend (with default backend) or just save to a file with agg
    
    # Returns
        None
    """
    if mapping_dict is None:
        mapping_dict = {-2: f"Too big >={limit_tokens}", -1:"Mixed answer", 0: "NON SEVERE", 1: "SEVERE"}
    if unique_values is None:
        unique_values = [-2,-1,0,1]
    # pretty print the confusion matrix
    values = [mapping_dict[e]+f"\n({e})" for e in unique_values]
    df_conf_matrix = pd.DataFrame(conf_matrix, index=values, columns=values)
    if backend is not None:
        matplotlib.use("agg")
    pp_matrix(df_conf_matrix, cmap="coolwarm", fz=11, figsize=[5,5], title="Confusion matrix\nBottom green recall;Right green precision\nBottom right accuracy\n",vmin=0,vmax=np.sum(conf_matrix))
    if folder_path is not None:
        plt.savefig(folder_path / "confusion_matrix.png")
    try:
        plt.show()
    except Exception as e:
        print(e)
        pass
def custom_encoder(obj):
    """Allow to encode into json file tuple keys"""
    return str(obj)

def find_representant(data: Union[Path,Dict[str,List[int]]], path_out: Path, poss: List[int], pred_field: str = "severity_pred", n_samples: int = 5, mapping_dict: Optional[Dict] = None, limit_tokens: int = 7366):
    """Find representants for each cell of the confusion matrix
    
    # Arguments
        - path_data: : Union[Path,Tuple[List[int],List[int]]], either 
            - Path, path to the folder where the json files where the data are (list of DataoutDict dicts)
            - Dict[str,List[int]], Dict with fields "bug_id", "binary_severity" "severity_pred" and "description" with binary_severity and severity_pred values with integers values only (no nan)
        - n_samples: the number of samples to get for each confusion matrix case
        - mapping_dict: Optional[Dict], mapping from possible values predicted to name (str or int), default {-2, "Too big query", -1:"Mixed answer", 0: "NON SEVERE", 1: "SEVERE"} and -2 replaces all nan
        - path_out: Path, path to where to store the representants
        - poss: the labels possibilities (-2 for nan)
    """
    if mapping_dict is None:
        mapping_dict = {-2: f"Too big >={limit_tokens}", -1:"Mixed answer", 0: "NON SEVERE", 1: "SEVERE"}
    if isinstance(data, Path) or isinstance(data, str):
        data = extract_fields_from_json(data, fields=["binary_severity",pred_field,"description"], allow_nan=True, allow_incoherent=True) #type: ignore
        
    samples = {}
    for i,j in product(poss,poss):
        samples[str((i,j))] = []
    for bug_id,true,pred,description in zip(data["bug_id"],data["binary_severity"],data[pred_field],data["description"]):
        pred = pred if not np.isnan(pred) else -2
        if len(samples[str((pred,true))]) < n_samples:
            samples[str((pred,true))].append({"bug_id":bug_id,"true":true,"pred":pred,"description":description})
        else:
            all_full = True
            for k in samples:
                all_full = all_full and (len(samples[k]) >= n_samples)
            if all_full:
                break
    for (true_pred),Lsamples in tqdm(samples.items(), total=len(samples)):
        true,pred =eval(true_pred)
        print("-"*10,f"true: {true} ; pred: {pred}","-"*10)
        for s in Lsamples:
            print("bug_id: ",s['bug_id'])
            print(s['description'])
    with open(path_out,"w") as f:
        json.dump({"samples":samples,"mapping":mapping_dict},f)
    return samples
    
class DataoutDict(TypedDict):
    bug_id: str
    text: str
    answer: str
    severity_pred: Union[int,float]
    binary_severity: int

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select LLM script to run")
    def path_check(p: str):
        path = Path(p)
        if path.exists():
            return path.resolve()
        else:
            raise Exception(f"{p} is not a path to a directory")
    algorithms_choices = ["inference","max_tokens", "compute_metrics", "explain", "finetune"]
    parser.add_argument("-path_data_json", type=path_check, help="Path to the json data file", default="/project/def-aloise/rmoine/data/data_preprocessed_tokens_v2.json")
    parser.add_argument("-path_data_folder", type=path_check, help="Root path to the main data folder", default="/project/def-aloise/rmoine/data/")
    parser.add_argument("-algorithm", choices=algorithms_choices, help="Algorithm to execute", default="inference")
    parser.add_argument("-token", choices=algorithms_choices, help="Token to huggingface", default="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")
    parser.add_argument("-interval_idx", type=int, help="Choice of the interval for inference and max tokens", default=0)
    parser.add_argument("-n_chunks", type=int, help="Number of chunks to do for inference and max tokens", default=3)
    args = parser.parse_args()
    print(args)
    n_data = 22302
    assert args.n_chunks is not None, "Expecting n_chunks"
    n_intervals = args.n_chunks
    intervals = [[i * (n_data // n_intervals), (i + 1) * (n_data // n_intervals)] for i in range(n_intervals)]
    if args.algorithm == "max_tokens":
        assert args.interval_idx is not None, "Expecting interval id"
        interval = intervals[args.interval_idx]
        get_max_tokens(args.path_data_json,token=args.token,start=interval[0],end=interval[1]+1)
    elif args.algorithm == "inference":
        assert args.interval_idx is not None, "Expecting interval id"
        interval = intervals[args.interval_idx]
        main_inference(args.path_data_json,token=args.token,start=interval[0],end=interval[1]+1,model_name="meta-llama/Llama-2-13b-chat-hf",id_pred="_trunc")
    elif args.algorithm == "compute_metrics":
        for pred_field,input_field in zip(["severity_pred","severity_pred2"],["text","trunc_text"]):
            path_out = args.path_data_folder / f"out_{pred_field}"
            path_out.mkdir(parents=True, exist_ok=True)
            compute_metrics(args.path_data_folder, path_out, pred_field=pred_field)
    elif args.algorithm == "explain":
        for pred_field,input_field in zip(["severity_pred","severity_pred2"],["text","trunc_text"]):
            path_out = args.path_data_folder / f"out_{pred_field}"
            path_in = Path(path_out / "representants.json")
            folder_out = path_in.parent / f"representants_{pred_field}_explain"
            main_shap(path_in, folder_out, token=args.token)
