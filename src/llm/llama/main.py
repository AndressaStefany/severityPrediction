import json
import numpy as np
import multiprocessing
from typing import *
import re
import transformers
try:
    import torch
except Exception:
    pass
from huggingface_hub import login
try:
    from nltk.tokenize import word_tokenize
except Exception:
    pass
from pathlib import Path
try:
    import pandas as pd
except Exception:
    pass
try:
    from src.baseline.baseline_functions import *
except Exception:
    pass
try:
    from rich.progress import track
except Exception:
    pass
try:
    from datasets import load_dataset
except Exception:
    pass
try:
    import matplotlib.pyplot as plt
    import matplotlib
except Exception:
    pass
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
except Exception:
    pass
import datetime
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        pipeline,
        BitsAndBytesConfig,
        BitsAndBytesConfig,
        HfArgumentParser,
        TrainingArguments,
        logging,
    )
except Exception:
    pass
try:
    from peft import LoraConfig, PeftModel
except Exception:
    pass
try:
    from trl import SFTTrainer
except Exception:
    pass
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
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def check_packages(*packages: str):
    def decorator(func):
        def inner_check(*args,**kwargs):
            missing = []
            for p in packages:
                if p not in sys.modules:
                    missing.append(p)
            if len(missing) > 0:
                raise Exception(f"{', '.join(missing)} are missing")
            
            return func(*args,**kwargs)
        return inner_check
    return decorator


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

def main_inference(path_descriptions: Path, model_name: str = "meta-llama/Llama-2-13b-chat-hf", token: str = "", start: int = 0, end: int = -1, limit_tokens: int = 7364):
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
    responses = []
    for i, d in tqdm(enumerate(data)):
        gc.collect()
        torch.cuda.empty_cache()
        answer = float('nan')
        severity = float('nan')
        text = d['text']
        n_tokens = len(tokenizer.tokenize(text))

        if n_tokens < limit_tokens:
            [answer] = pipeline(
                text,
                do_sample=True,
                top_k=1,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=1024,
                return_full_text=False
            )
            answer = answer['generated_text']
            if not isinstance(answer, str):
                raise Exception("Unknown result answer: "+str(answer))
            severity = classify(answer)

        responses.append({**d, "answer": answer, "severity_pred": severity})
        if i%5==0:
            with open(path_descriptions.parent / f"predictions/predictions_v100l_chunk_{start}.json", "w") as f:
                json.dump(responses,f)
        
    with open(path_descriptions.parent / f"predictions/predictions_v100l_chunk_{start}.json", "w") as f:
        json.dump(responses,f)

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

def main_qlora(new_model_name: str, file_examples: Path, folder_out: Path, model_name: str = "meta-llama/Llama-2-13b-chat-hf", token: str = "",
               lora_alpha: float = 16, lora_dropout: float = 0.1, lora_r: int = 64, 
               num_train_epochs: int = 1, tr_bs: int = 4, val_bs: int = 4,
               optim: str = "paged_adamw_32bit", save_steps: int = 25,
               logging_steps: int = 25, learning_rate: float = 2e-4, weight_decay: float = 0.001, 
               fp16: bool = False, bf16: bool = False, max_grad_norm: float = 0.3, max_steps: int = -1, warmup_ratio: float = 0.03,
               group_by_length: bool = True, lr_scheduler_type: str = "constant", eval_steps: int = 20,
               train_size: float = 0.3
               ):
    """
    Perform training and fine-tuning of a model for causal reasoning using LoRA.
    Doc: https://miro.medium.com/v2/resize:fit:4800/format:webp/1*rOW5plKBuMlGgpD0SO8nZA.png

    # Arguments
        - new_model_name: str, name of the new model pretrained
        - file_examples: Path, a file path to input data.
        - folder_out: Path, a Path object representing the output folder for the results.
        - model_name: str, the name or path of the pretrained model to use. Default: "meta-llama/Llama-2-13b-chat-hf"
        - token: str, a token string. Default: ""
        - lora_alpha: float, scaling factor for the weight matrices. alpha is a scaling factor that adjusts the magnitude of the combined result (base model output + low-rank adaptation). Default: 16
        - lora_dropout: float, dropout probability of the LoRA layers. This parameter is used to avoid overfitting. Default: 0.1
        - lora_r: int, this is the dimension of the low-rank matrix. Default: 64. It means for a layer initialy of size d_in x d_out we will have 2 lora layers of size d_in x r and r x d_out reducing the number of parameters
        - num_train_epochs: int, the number of training epochs. Default: 1
        - tr_bs: int, batch size for training. Default: 4
        - val_bs: int, batch size for validation. Default: 4
        - optim: str, optimization method. Possible values include "paged_adamw_32bit" and other optimization methods specific to the project. Default: "paged_adamw_32bit"
        - save_steps: int, the frequency of saving model checkpoints during training. Default: 25
        - logging_steps: int, the frequency of logging training progress. Default: 25
        - learning_rate: float, the learning rate for training. Default: 2e-4
        - weight_decay: float, the weight decay value for regularization. Default: 0.001
        - fp16: bool, whether to use mixed-precision training with 16-bit floats. Default: False
        - bf16: bool, whether to use 16-bit bfloat16 format. Default: False
        - max_grad_norm: float, the maximum gradient norm for gradient clipping. Default: 0.3
        - max_steps: int, the maximum number of training steps. Default: -1 (unlimited)
        - warmup_ratio: float, the warmup ratio for learning rate scheduling. Default: 0.03
        - group_by_length: bool, a flag to group data by sequence length. Default: True
        - lr_scheduler_type: str, type of learning rate scheduler. Default: "constant"
        - eval_steps: int, the frequency of evaluating the model during training. Default: 20
        - train_size: float = 0.3, the size of the training dataset
    """
    print("main_qlora")
    if not folder_out.exists():
        folder_out.mkdir(parents=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    double_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )
    n_gpus = torch.cuda.device_count() #type: ignore
    max_memory = f'{32}GB'
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=double_quant_config,
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    peft_config = LoraConfig( #type: ignore
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=str(folder_out.resolve()),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=tr_bs,
        gradient_accumulation_steps=val_bs,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="all", #type: ignore
        evaluation_strategy="steps",
        eval_steps=eval_steps
    )
    # create datasets
    print("Create datasets")
    train_path = folder_out / "train.json"
    valid_path = folder_out / "valid.json"
    if not train_path.exists() or not valid_path.exists():
        with open(file_examples, "r") as f:
            d = json.load(f)
        idx_tr, idx_val = train_test_split(np.arange(len(d)), train_size=train_size)
        idx_tr = set(idx_tr)
        idx_val = set(idx_val)
        with open(train_path, "w") as f:
            json.dump([e for i,e in enumerate(d) if i in idx_tr],f)
        with open(valid_path, "w") as f:
            json.dump([e for i,e in enumerate(d) if i in idx_val],f)
    train_dataset = load_dataset('json', data_files=str(train_path.resolve()), split="train") #type: ignore
    valid_dataset = load_dataset('json', data_files=str(valid_path.resolve()), split="train") #type: ignore
    extract_in_out = lambda examples: {'text': examples['trunc_text']}
    train_dataset = train_dataset.map(extract_in_out, batched=True)
    valid_dataset = valid_dataset.map(extract_in_out, batched=True)
    # Set supervised fine-tuning parameters
    print("Set supervised fine-tuning parameters")
    trainer = SFTTrainer( #type: ignore
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,  # Pass validation dataset here
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )
    print("Starting training QLORA")
    trainer.train()
    print("Saving trained QLORA model")
    trainer.model.save_pretrained(new_model_name)
    
    
@check_packages("torch", "shap")
def main_shap(file_examples: Path, folder_out: Path, model_name: str = "meta-llama/Llama-2-13b-chat-hf", token: str = ""):
    print("main_shap")
    if not folder_out.exists():
        folder_out.mkdir(parents=True)
    # open sentences
    with open(file_examples) as f:
        representants = json.load(f)
        representants = {eval(k):v for k,v in representants['samples'].items()} 
    print("data loaded")
    if token != "":
        login(token=token)
    print("loading model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    double_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
    )
    n_gpus = torch.cuda.device_count() #type: ignore
    max_memory = f'{32}GB'
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=double_quant_config,
        max_memory = {i: max_memory for i in range(n_gpus)},
        device_map="auto"
    )
    gen_dict = dict(
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
        return_full_text=False
    )
    model.config.task_specific_params = dict()
    model.config.task_specific_params["text-generation"] = gen_dict
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"   
    )
    teacher_forcing_model = shap.models.TeacherForcing(model, tokenizer) #type: ignore
    masker = shap.maskers.Text(tokenizer, mask_token="...", collapse_mask_token=True) #type: ignore
    explainer = shap.Explainer(teacher_forcing_model, masker) #type: ignore
    # explainer = shap.Explainer(model, tokenizer) #type: ignore
    print("Starting inference")
    for k,Ltext in representants.items():
        for i,text in enumerate(Ltext):
            gc.collect()
            torch.cuda.empty_cache()
            text = text['input']
            x = [text,text]
            y = ["0","1"]
            print("Prediction for ")
            print(text)
            shap_values = explainer(x,y)
            with open(folder_out / f"shap_pred_{k[0]}_true_{k[1]}_sample_{i}.html","w") as f:
                f.write(shap.plots.text(shap_values, display=False)) #type: ignore
def compute_metrics(folder_predictions: Path, folder_out: Optional[Path] = None, input_text_field: str = "text", pred_field: str = "severity_pred", mapping_dict: Optional[dict] = None, limit_tokens: int = 7364):
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
    fields_data = extract_fields_from_json(folder_predictions, fields=["bug_id", "binary_severity", pred_field, "description",input_text_field], allow_nan=True, allow_incoherent=True)
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
        pred_field=pred_field,
        input_text_field=input_text_field
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

def find_representant(data: Union[Path,Dict[str,List[int]]], path_out: Path, poss: List[int], input_text_field: str, pred_field: str = "severity_pred", n_samples: int = 5, mapping_dict: Optional[Dict] = None, limit_tokens: int = 7366):
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
        data = extract_fields_from_json(data, fields=["binary_severity",pred_field,"description",input_text_field], allow_nan=True, allow_incoherent=True) #type: ignore
        
    samples = {}
    for i,j in product(poss,poss):
        samples[str((i,j))] = []
    for bug_id,true,pred,description,input_field in zip(data["bug_id"],data["binary_severity"],data[pred_field],data["description"],data[input_text_field]):
        pred = pred if not np.isnan(pred) else -2
        
        if len(samples[str((pred,true))]) < n_samples and len(description.strip())>0:
            samples[str((pred,true))].append({"bug_id":bug_id,"true":true,"pred":pred,"description":description,"input":input_field})
            samples[str((pred,true))].sort(key=lambda x:len(x["description"]), reverse=True)
        elif len(samples[str((pred,true))]) >= n_samples and len(samples[str((pred,true))][0]["description"])>len(description) and len(description.strip()) > 0:
            samples[str((pred,true))].pop(0)
            samples[str((pred,true))].append({"bug_id":bug_id,"true":true,"pred":pred,"description":description,"input":input_field})
            samples[str((pred,true))].sort(key=lambda x:len(x["description"]), reverse=True)
            
            
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
    # logging.basicConfig(filename='/project/def-aloise/rmoine/log-severity.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger('severity')
    # main("TinyPixel/Llama-2-7B-bf16-sharded")
    # preprocess_data("eclipse_clear.json", Path("data"),few_shots=False,id="")
    # preprocess_data("eclipse_clear.json", Path("data"),few_shots=True,id="_few_shots")
    # path_data = Path("/project/def-aloise/andressa/eclipse_with_text.json")
    path_data = Path("/project/def-aloise/rmoine/data/predictions/")# /project/def-aloise/rmoine
    
    # get_max_tokens(path_data,token="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",start=0,end=24225)
    # get_max_tokens(path_data,token="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",start=24225,end=48450)
    # get_max_tokens(path_data,token="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",start=48450,end=72676)
    # main(path_data,token="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",start=0,end=24225)
    # main(path_data,token="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",start=24225,end=48450)
    # main(path_data,token="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",start=48450,end=72676)
    # for pred_field,input_field in zip(["severity_pred","severity_pred2"],["text","trunc_text"]):
    #     path_out = path_data / f"out_{pred_field}"
    #     path_out.mkdir(parents=True, exist_ok=True)
    #     compute_metrics(path_data, path_out, pred_field=pred_field,input_text_field=input_field)
    
    #     path_in = Path(path_out / "representants.json")
    #     folder_out = path_in.parent / f"representants_{pred_field}_explain"
    #     # main_shap(path_in, folder_out, token="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")
    path_examples = path_data / "predictions_v100l_chunk_0.json"
    folder_out = path_data.parent.parent / "out_qlora"
    folder_out.mkdir(exist_ok=True)
    main_qlora("llama-13b-finetune",path_examples,folder_out,token="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")
