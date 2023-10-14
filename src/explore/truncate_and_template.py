import json
from pathlib import Path
from tqdm import tqdm

from string import Template 

from huggingface_hub import login

from transformers import (
    AutoTokenizer,
)

PREPROMPT = 'Always answer with one token. Do not give any explanation. Use only 0 or 1 and one token. Skip any politeness answer. You have only one word available.\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.'
INSTRUCTIONS = 'Categorize the bug report into one of 2 categories:\n\n0 = NOT SEVERE\n1 = SEVERE\n'


def get_template(preprompt = PREPROMPT, instructions = INSTRUCTIONS, description = ""):
    """
    This function generates a template for text based on the provided input strings.

    Args:
        preprompt (str): The preprompt text.
        instructions (str): The instructions text.
        description (str): The description text to be inserted into the template.

    Returns:
        str: A template string with placeholders replaced by the provided input strings.
    """
    t = Template('$preprompt\n\n### Instruction:\n$add_instructions\n### Input:\n$input\n\n### Remembering the instruction:\n$last_question\n### Response:')

    return t.substitute({'preprompt': preprompt,
                        'add_instructions': instructions, 
                        'input': description,
                        'last_question': instructions})


def truncate_and_transform(data, model_name = "meta-llama/Llama-2-13b-chat-hf", token = ""):
    """
    This function takes a list of dictionaries, where each dictionary contains a 'description' field.
    It truncates the 'description' text using the specified model's tokenizer, and then transforms 
    the truncated text into a 'trunc_description' field. Additionally, it transforms the truncated 
    text into a 'trunc_tex' field using the 'get_template' function.
    
    Args:
        data (list of dict): A list of dictionaries where each dictionary contains
                            a 'description' field with text to be processed.
        model_name (str): The name of the pretrained model for tokenization.

    Returns:
        None: The function modifies the input 'data' dictionaries in place by adding 
              'trunc_description' and 'trunc_tex' fields.
    """
    if token != "":
        login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set the padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
    for i, d in tqdm(enumerate(data)):
        print("Tokenizing",d)
        tokenized_desc = tokenizer.encode_plus(
            d['description'],
            max_length=7000,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        truncated_token = tokenized_desc["input_ids"]
        truncated_text = tokenizer.decode(truncated_token[0], skip_special_tokens=True)
        
        d['trunc_description'] = truncated_text
        d['truncated_token'] = [t for t in tokenizer.convert_ids_to_tokens(truncated_token[0]) if t != "[PAD]"]
        d['trunc_text'] = get_template(description = truncated_text)


if __name__ == "__main__":
    path_data = Path('../../predictions/chunck/predictions_v100l_all_chunks.json')
    with open(path_data) as f:
        data = json.load(f)
        
    model = "meta-llama/Llama-2-13b-chat-hf"
    token = "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"
    truncate_and_transform(data=data, model_name=model, token=token)
    
    updated_json = json.dumps(data, indent=4)
    with open(path_data.parent/'chuncks_trunc.json', 'w') as file:
        file.write(updated_json)
