from transformers import AutoTokenizer
import transformers
import torch
from huggingface_hub import login
import os

def main(model: str = "meta-llama/Llama-2-7b-chat-hf", token: str = "hf_xxx"):
    if token == "hf_xxx":
        login(token=token)
    else:
        token = token
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
        
if __name__ == "_main__":
    main("TinyPixel/Llama-2-7B-bf16-sharded")