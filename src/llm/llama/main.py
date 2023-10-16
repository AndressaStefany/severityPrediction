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
try:
    from peft import LoraConfig, PeftModel
except Exception:
    pass
try:
    from trl import SFTTrainer
except Exception:
    pass
import json
import numpy as np
import multiprocessing
from typing import *
import re
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
)
import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    TrainingArguments,
)
import gc
import os
import math
from tqdm import tqdm
try:
    from pretty_confusion_matrix import pp_matrix
except Exception:
    try:
        from src.llm.llama.pretty_confusion_matrix import pp_matrix
    except Exception:
        pass
from itertools import product
from textwrap import wrap

try:
    import shap
except Exception:
    pass
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def print_args(func):
    def inner(*args, **kwargs):
        print("*" * 100)
        print("Start", func.__name__)
        print("With *args", args)
        print("With **kwargs", kwargs)
        print("-" * 100)
        return func(*args, **kwargs)

    return inner


def classify(answer: str) -> int:
    """Return 0 if not severe, 1 if severe and -1 if unknown"""
    pattern_severe = "[sS][eE][vV][eE][rR][eE]"
    pattern_not_severe = "[nN][oO][tT] *" + pattern_severe
    if re.match(pattern_not_severe, answer) is not None or (
        "0" in answer and "1" not in answer
    ):
        return 0
    elif re.match(pattern_severe, answer) is not None or (
        "1" in answer and "0" not in answer
    ):
        return 1
    return -1


def get_tokens(data, tokenizer):
    Ltokens = []
    for i, d in tqdm(enumerate(data)):
        gc.collect()
        torch.cuda.empty_cache()
        text = d["text"]
        n_tokens = len(tokenizer.tokenize(text))
        Ltokens.append(n_tokens)
    return Ltokens


def get_max_mix(token_lengths, tokenizer, pipeline):
    min_token_length = min(token_lengths)
    max_token_length = max(token_lengths)

    max_work = 0
    min_not_work = float("inf")

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


def get_max_tokens(
    path_descriptions: Path,
    model_name: str = "meta-llama/Llama-2-13b-chat-hf",
    token: str = "",
    start: int = 0,
    end: int = -1,
):
    print("Start")
    if token != "":
        login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    double_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=double_quant_config
    )
    pipeline = transformers.pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
    )
    with open(path_descriptions) as f:
        data = json.load(f)
    if end == -1:
        end = len(data)
    data = data[start:end]

    token_lengths = get_tokens(data, tokenizer)
    (max_work, min_not_work) = get_max_mix(token_lengths, tokenizer, pipeline)
    with open(
        path_descriptions.parent / f"max_tokens_v100l_chunk_{start}.json", "w"
    ) as f:
        json.dump(
            {
                "min_not_work": min_not_work,
                "max_work": max_work,
                "number_of_tokens": token_lengths,
            },
            f,
        )


def build_prompt(
    llama_tokenized_template: List[str],
    llama_tokenized_description: List[str],
    template_index_insert: int,
    tokenizer,
    limit_tokens: int
) -> Tuple[str, List[str]]:
    """We put the template and the description together using the insertion point specified in description_index_insert"""
    ## First make a copy
    tokenized_full_text = llama_tokenized_template[:]
    ## Then remove the <s> token from the description
    description = llama_tokenized_description[1:]
    # limit the number of tokens of the description
    n_tokens_descr_max = limit_tokens-len(tokenized_full_text)
    description = description[:n_tokens_descr_max]
    ## Then insert the description inside the template at the position indicated (after input)
    tokenized_full_text[template_index_insert:template_index_insert] = description
    ## And remove the start token (as it will be put again by the tokenizer)
    tokenized_full_text.pop(0)
    ## Convert back into a sentence
    text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenized_full_text))
    return text, tokenized_full_text


@print_args
def main_inference(
    path_data_preprocessed: Path,
    model_name: str = "meta-llama/Llama-2-13b-chat-hf",
    token: str = "",
    start: int = 0,
    end: int = -1,
    limit_tokens: int = 7364,
    id_pred: str = "",
):
    if token != "":
        login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    double_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_type=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=double_quant_config
    )
    pipeline = transformers.pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
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
    folder_predictions.mkdir(exist_ok=True, parents=True)
    with open(folder_predictions / f"metadata.meta", "w") as f:
        json.dump({"data_path":str(path_data_preprocessed.resolve())}, f, indent=2)
    for i, d in tqdm(enumerate(data), total=len(data)):
        # gc.collect()
        # torch.cuda.empty_cache()  # type: ignore
        answer = float("nan")
        severity = float("nan")
        text, tokenized_full_text = build_prompt(
            template["llama_tokenized_template"],
            d["llama_tokenized_description"],
            template["template_index_insert"],
            tokenizer,
            limit_tokens=limit_tokens
        )
        n_tokens = len(tokenized_full_text)
        if n_tokens < limit_tokens:
            try:
                [answer] = pipeline(  # type: ignore
                    text,
                    do_sample=True,
                    top_k=1,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False,
                )
                answer = answer["generated_text"]  # type: ignore
                if not isinstance(answer, str):
                    raise Exception("Unknown result answer: " + str(answer))
                severity = classify(answer)
            except Exception:
                severity = -2

        responses.append(
            {
                **d,
                "answer": answer,
                f"severity_pred{id_pred}": severity,
                f"input{id_pred}": text,
            }
        )
        if i % 5 == 0:
            with open(
                folder_predictions / f"predictions_v100l_chunk_{start}_{id_pred}.json", "w"
            ) as f:
                json.dump(responses, f)

    with open(folder_predictions / f"predictions_v100l_chunk_{start}_{id_pred}.json", "w") as f:
        json.dump(responses, f, indent=2)


def extract_fields_from_json(
    folder_path: Path
) -> List[Dict]:
    """Aggregates every predictions and true value stored in each json file stored in folder_path with possibility to choose which fields to return

    # Arguments
        - folder_path: Path, the path where all of the *.json files are stored. They must contain List[dict] with in dict keys 'binary_severity' and 'severity_pred'

    # Returns
        - List[Dict], Lists for each field required in fields associated to their field name
    """

    folder_path = Path(folder_path)
    json_file_paths = [file for file in folder_path.glob("*.json")]
    L = []
    for json_file_path in json_file_paths:
        with open(json_file_path) as f:
            data: List[DataoutDict] = json.load(f)
        L.extend(data)
    return L


def main_shap(
    file_examples: Path,
    folder_out: Path,
    model_name: str = "meta-llama/Llama-2-13b-chat-hf",
    token: str = "",
):
    if not folder_out.exists():
        folder_out.mkdir(parents=True)
    # open sentences
    with open(file_examples) as f:
        representants = json.load(f)
        mapping = representants["mapping"]
        representants = {eval(k): v for k, v in representants["samples"].items()}
    if token != "":
        login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    double_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=double_quant_config
    )
    pipeline = transformers.pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
    )
    pipeline.config.task_specific_params["text-generation"] = dict(
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
        return_full_text=False,
    )
    print("Starting inference")
    for k, Ltext in representants.items():
        for i, text in enumerate(Ltext):
            explainer = shap.Explainer(pipeline, tokenizer)
            shap_values = explainer([text])
            with open(
                folder_out / f"shap_pred_{k[0]}_true_{k[1]}_sample_{i}.html", "w"
            ) as f:
                f.write(shap.plots.text(shap_values, display=False))


def compute_metrics(
    folder_predictions: Path,
    folder_out: Optional[Path] = None,
    input_field: str = "input",
    pred_field: str = "severity_pred",
    mapping_dict: Optional[dict] = None,
    n_tokens_infered_max: int = 7364,
    n_tokens_show_max: int = 7364,
    n_tokens_show_min: int = 0,
    model_name: str = "meta-llama/Llama-2-13b-chat-hf",
    token: str = "",
):
    """Taking the path of the predictions folder, it computes the statistics with the predictions (confusion matrix, precision, recall, f1-score). The confusion matrix is plotted into a png file

    # Arguments
        - folder_predictions: Path, path to the folder where the prediction files are stored
        - folder_out: Path, path to the folder where the statistics will be stored
        - mapping_dict: Optional[Dict], mapping from possible values predicted to name (str or int), default {-2:"Too big query", -1:"Mixed answer", 0: "NON SEVERE", 1: "SEVERE"} and -2 replaces all nan

    # Return
        None
    """
    if token != "":
        login(token=token)
    if folder_out is None:
        folder_out = folder_predictions
    fields_data = extract_fields_from_json(
        folder_predictions
    )
    data = pd.DataFrame(fields_data)
    # Count number of tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    data["n_tokens"] = data[input_field].apply(lambda x: len(tokenizer(x)['input_ids']))#type: ignore
    # Filter by limit of tokens
    data = data.query(
        f"n_tokens < {n_tokens_show_max} & n_tokens >= {n_tokens_show_min}"
    )
    # Replace Nan by -2 in prediction
    data[pred_field] = data[pred_field].apply(lambda x: -2 if np.isnan(x) else int(x))
    data["binary_severity"] = np.where(
        data[pred_field] == -2, -2, data["binary_severity"]
    )
    data.rename({"binary_severity": "true", pred_field: "pred"}, axis=1, inplace=True)
    # Apply mapping
    if mapping_dict is None:
        mapping_dict = {
            -2: f"Too big >={n_tokens_infered_max}",
            -1: "Mixed answer",
            0: "NON SEVERE",
            1: "SEVERE",
        }
    data["pred_text"] = data["pred"].apply(lambda x: mapping_dict[x])
    data["true_text"] = data["true"].apply(lambda x: mapping_dict[x])
    true = np.array(data["true"])
    pred = np.array(data["pred"])
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true, pred)
    # Compute accuracy
    accuracy = accuracy_score(true, pred)
    # Compute precision
    precision: np.ndarray = precision_score(true, pred, average=None)  # type: ignore

    # Compute recall
    recall: np.ndarray = recall_score(true, pred, average=None)  # type: ignore

    # Compute F1-score
    f1: np.ndarray = f1_score(true, pred, average=None)  # type: ignore
    data.to_json(folder_out / "data.json",orient="records",indent=4)
    with open(folder_out / "metrics.json", "w") as f:
        json.dump(
            {
                "date_timestamp": datetime.datetime.now().timestamp(),
                "confusion_matrix": conf_matrix.tolist(),
                "accuracy": accuracy,
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "f1": f1.tolist(),
            },
            f,
        )
    possibilities_pred = sorted(list(set(data["pred"].unique().tolist()).union(set(data["true"].unique().tolist()))))
    id = f"_field_{pred_field}_shown_{n_tokens_show_min}_{n_tokens_show_max}_trunc_{n_tokens_infered_max}"
    plot_confusion(
        conf_matrix=conf_matrix,
        folder_path=folder_out,
        mapping_dict=mapping_dict,
        unique_values=possibilities_pred,
        limit_tokens=n_tokens_infered_max,
        backend="agg",  # type: ignore
        title=f"Confusion matrix\nfor field {pred_field}\n{n_tokens_infered_max=}\nn_tokens_shown in [{n_tokens_show_min};{n_tokens_show_max}[",
        id=id,
    )
    # find representants
    find_representant(
        data,
        folder_out / f"representants{id}.json",
        n_samples=5,
        mapping_dict=mapping_dict,
    )


def plot_confusion(
    conf_matrix: np.ndarray,
    folder_path: Optional[Path] = None,
    mapping_dict: Optional[Dict] = None,
    unique_values: Optional[List] = None,
    backend=Optional[Literal["agg"]],
    limit_tokens: int = 7366,
    title: str = "Confusion matrix",
    id: str = "",
):
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
        mapping_dict = {
            -2: f"Too big >={limit_tokens}",
            -1: "Mixed answer",
            0: "NON SEVERE",
            1: "SEVERE",
        }
    if unique_values is None:
        unique_values = [-2, -1, 0, 1]
    # pretty print the confusion matrix
    values = [mapping_dict[e] + f"\n({e})" for e in unique_values]
    df_conf_matrix = pd.DataFrame(conf_matrix, index=values, columns=values)
    if backend is not None:
        matplotlib.use("agg")
    pp_matrix(# type: ignor
        df_conf_matrix,
        cmap="coolwarm",
        fz=11,
        figsize=[5, 5],
        title=title,
        vmin=0,
        vmax=np.sum(conf_matrix),
    )
    if folder_path is not None:
        plt.savefig(folder_path / f"confusion_matrix{id}.png")
    try:
        plt.show()
    except Exception as e:
        print(e)
        pass


def custom_encoder(obj):
    """Allow to encode into json file tuple keys"""
    return str(obj)


def find_representant(
    df: pd.DataFrame, path_out: Path, mapping_dict: Dict, n_samples: int = 5
):
    """Find representants for each cell of the confusion matrix

    # Arguments
        - df: : dataframe containing in pred the integer prediction (no nan) and in true the integer true value
        - n_samples: the number of samples to get for each confusion matrix case
        - mapping_dict: Optional[Dict], mapping from possible values predicted to name (str or int), default {-2, "Too big query", -1:"Mixed answer", 0: "NON SEVERE", 1: "SEVERE"} and -2 replaces all nan
        - path_out: Path, path to where to store the representants
        - poss: the labels possibilities (-2 for nan)
    """
    samples = {}
    poss = df["pred"].unique().tolist()
    for i, j in product(poss, poss):
        df_sel = (
            df.query(f"pred == {i} & true == {j}")
            .head(n_samples)
            .to_dict(orient="records")
        )
        samples[str((i, j))] = df_sel
    with open(path_out, "w") as f:
        json.dump({"samples": samples, "mapping": mapping_dict}, f)
    return samples

@print_args
def main_qlora(
    new_model_name: str, file_examples: Path, folder_out: Path, model_name: str = "meta-llama/Llama-2-13b-chat-hf", token: str = "", field_input: str = "trunc_text", field_label: str  = "binary_severity",
               input_field: str = "trunc_text",
               lora_alpha: float = 16, lora_dropout: float = 0.1, lora_r: int = 64, 
               num_train_epochs: int = 1, tr_bs: int = 4, val_bs: int = 4,
               optim: str = "paged_adamw_32bit", save_steps: int = 25,
               logging_steps: int = 25, learning_rate: float = 2e-4, weight_decay: float = 0.001, 
               fp16: bool = False, bf16: bool = False, max_grad_norm: float = 0.3, max_steps: int = -1, warmup_ratio: float = 0.03,
               group_by_length: bool = True, lr_scheduler_type: str = "constant", eval_steps: int = 20,
               train_size: float = 0.3, limit_tokens: int = 7364,
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
    if token != "":
        login(token=token)
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
    train_path = folder_out / f"train_max_{limit_tokens}.json"
    valid_path = folder_out / f"valid_max_{limit_tokens}.json"
    if not train_path.exists() or not valid_path.exists():
        with open(file_examples) as f:
            data_preprocessed = json.load(f)
        data = data_preprocessed["data"]
        template = data_preprocessed["template"]
        L = []
        for d in data:
            text, tokenized_full_text = build_prompt(
                template["llama_tokenized_template"],
                d["llama_tokenized_description"],
                template["template_index_insert"],
                tokenizer,
                limit_tokens=limit_tokens
            )
            L.append({**d,"text":text,"tokenized_full_text":tokenized_full_text})
        idx_tr, idx_val = train_test_split(np.arange(len(L)), train_size=train_size) #type: ignore
        idx_tr = set(idx_tr)
        idx_val = set(idx_val)
        with open(train_path, "w") as f:
            json.dump([{"text":e["text"]} for i,e in enumerate(L) if i in idx_tr],f)
        with open(valid_path, "w") as f:
            json.dump([{"text":e["text"]} for i,e in enumerate(L) if i in idx_val],f)
    train_dataset = load_dataset('json', data_files=str(train_path.resolve()), split="train") #type: ignore
    valid_dataset = load_dataset('json', data_files=str(valid_path.resolve()), split="train") #type: ignore
    # Set supervised fine-tuning parameters
    print("Set supervised fine-tuning parameters")
    trainer = SFTTrainer( #type: ignore
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,  # Pass validation dataset here
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )
    print("Starting training QLORA")
    trainer.train()
    print("Saving trained QLORA model")
    trainer.model.save_pretrained(new_model_name)

class DataoutDict(TypedDict):
    bug_id: str
    text: str
    answer: str
    severity_pred: Union[int, float]
    binary_severity: int


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select LLM script to run")

    def path_check(p: str):
        path = Path(p)
        if path.exists():
            return path.resolve()
        else:
            raise Exception(f"{p} is not a path to a directory")

    algorithms_choices = [
        "inference",
        "max_tokens",
        "compute_metrics",
        "explain",
        "finetune",
    ]
    parser.add_argument(
        "-path_data_json",
        type=path_check,
        help="Path to the json data file",
        default="/project/def-aloise/rmoine/data/data_preprocessed_tokens_v2.json",
    )
    parser.add_argument(
        "-path_data_folder",
        type=path_check,
        help="Root path to the main data folder",
        default="/project/def-aloise/rmoine/data/",
    )
    parser.add_argument(
        "-algorithm",
        choices=algorithms_choices,
        help="Algorithm to execute",
        default="inference",
    )
    parser.add_argument(
        "-token",
        choices=algorithms_choices,
        help="Token to huggingface",
        default="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",
    )
    parser.add_argument(
        "-interval_idx",
        type=int,
        help="Choice of the interval for inference and max tokens",
    )
    parser.add_argument(
        "-n_chunks",
        type=int,
        help="Number of chunks to do for inference and max tokens",
    )
    parser.add_argument(
        "-seed_start", type=int, help="Seed start for inference", default=0
    )
    parser.add_argument(
        "-seed_end", type=int, help="Seed end for inference (included)", default=-1
    )
    parser.add_argument(
        "-pred_field", type=str, help="Predicted field to use", default="severity_pred_trunc"
    )
    parser.add_argument(
        "-input_field", type=str, help="Predicted field to use", default="description"
    )
    parser.add_argument(
        "-n_tokens_infered_max", type=int, help="Number of maximum infered tokens", default=7364
    )
    parser.add_argument(
        "-n_tokens_show_min", type=int, help="Maximum number of tokens tokens in the statistics", default=0
    )
    parser.add_argument(
        "-n_tokens_show_max", type=int, help="Minimum number of tokens shown in the statistics", default=7364
    )
    parser.add_argument(
        "-n_data", type=int, help="Total number of data in the dataset", default=22302
    )
    parser.add_argument(
        "-id", type=str, help="Id to put on the files to save", default="_trunc"
    )
    args = parser.parse_args()
    print(args)
    n_data = args.n_data
    assert args.n_chunks is not None or (
        args.seed_start is not None and args.seed_end is not None
    ), "Expecting n_chunks or seed start and seed end"
    if args.n_chunks is not None:
        assert args.interval_idx is not None, "Expecting interval id"
        assert args.interval_idx is not None, "Expecting interval id"

        n_intervals = args.n_chunks
        intervals = [
            [i * (n_data // n_intervals), (i + 1) * (n_data // n_intervals)]
            for i in range(n_intervals)
        ]
        intervals[-1][1] += 1
        [seed_start, seed_end] = intervals[args.interval_idx]
    else:
        [seed_start, seed_end] = [args.seed_start, args.seed_end]
    if args.algorithm == "max_tokens":
        get_max_tokens(
            args.path_data_json, token=args.token, start=seed_start, end=seed_end
        )
    elif args.algorithm == "inference":
        main_inference(
            args.path_data_json,
            token=args.token,
            start=seed_start,
            end=seed_end,
            model_name="meta-llama/Llama-2-13b-chat-hf",
            id_pred=args.id,
        )
    elif args.algorithm == "compute_metrics":
        path_out = args.path_data_folder / f"out_{args.pred_field}"
        path_out.mkdir(parents=True, exist_ok=True)
        compute_metrics(
            args.path_data_folder,
            path_out,
            pred_field=args.pred_field,
            input_field=args.input_field,
            token=args.token,
            n_tokens_infered_max=args.n_tokens_infered_max,
            n_tokens_show_min=args.n_tokens_show_min,
            n_tokens_show_max=args.n_tokens_show_max,
        )
    elif args.algorithm == "explain":
        for pred_field, input_field in zip(
            ["severity_pred", "severity_pred2"], ["text", "trunc_text"]
        ):
            path_out = args.path_data_folder / f"out_{pred_field}"
            path_in = Path(path_out / "representants.json")
            folder_out = path_in.parent / f"representants_{pred_field}_explain"
            main_shap(path_in, folder_out, token=args.token)