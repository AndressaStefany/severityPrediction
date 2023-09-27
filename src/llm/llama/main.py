from transformers import AutoTokenizer
import transformers
import torch
from huggingface_hub import login
import logging
from nltk.tokenize import word_tokenize
from pathlib import Path
import pandas as pd



    
    
    
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
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200, # I'll change here
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
        
if __name__ == "__main__":
    # logging.basicConfig(filename='/project/def-aloise/rmoine/log-severity.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger('severity')
    main("TinyPixel/Llama-2-7B-bf16-sharded")