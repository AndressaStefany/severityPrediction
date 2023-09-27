from transformers import AutoTokenizer
import transformers
# import torch
from huggingface_hub import login
import logging
from nltk.tokenize import word_tokenize
from pathlib import Path
import pandas as pd
from src.baseline.baseline_functions import *
import json
import multiprocessing
from typing import *

def process(l: str) -> Optional[dict]:
    """The multiprocessing function that generates the dictionnary from a line of the eclipse_clear.json file"""
    global default_severities_to_keep
    global default_high_severities_vals
    data: dict = eval(l.strip())
    # filter severities
    if data['bug_severity'] not in default_severities_to_keep:
        return
    # binarize severities
    severity = 1 if data['bug_severity'] in default_high_severities_vals else 0
    # process descriptions
    description = data['description']
    if isinstance(description, list):
        description = " ".join(description)
    if not isinstance(description, str):
        return 
    description = description.strip()
    if description == "":
        return
    description = remove_url(description)
    description = build_prompt(description)
    _id = data['_id']
    bug_id = data['bug_id']
    return {"_id": _id, "bug_id": bug_id, "severity": severity, "description": description}

def preprocess_data(file_name: str, data_folder: Path):
    """Takes the csv file as input, apply the preprocessings and write the resulting data to files"""
    print("Starting preprocessing")
    # open json file
    with open(data_folder / file_name) as f:
        data: List[str] = f.readlines()
    with multiprocessing.Pool() as p:
        data_processed: List[dict] = [e for e in p.map(process, data) if e is not None]
    for i in range(len(data_processed)):
        data_processed[i]['idx'] = i
    data_out = {key: [d[key] for d in data_processed] for key in data_processed[0]}
    # write to a file
    folder = data_folder / "llm"
    folder.mkdir(exist_ok=True)
    print("saving...")
    ## save the description in one dedicated json file (list of str)
    with open(folder / "llm_descriptions.json", "w") as f:
        json.dump(data_out['description'], f, indent=2)
    ## save everything in a metadata file
    with open(folder / "llm_full_data.json", "w") as f:
        json.dump(data_out, f, indent=2)
def main(model: str = "meta-llama/Llama-2-7b-chat-hf", token: str = ""):
    print('Start')
    if token != "":
        login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200, # I'll change here
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
        
if __name__ == "__main__":
    import warnings

    # Ignore DeprecationWarning
    warnings.filterwarnings("ignore")
    # logging.basicConfig(filename='/project/def-aloise/rmoine/log-severity.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger('severity')
    # main("TinyPixel/Llama-2-7B-bf16-sharded")
    preprocess_data("eclipse_clear.json", Path("data"))