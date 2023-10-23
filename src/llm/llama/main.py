
from pathlib import Path
from typing import *#type: ignore
import json
import re
import datetime
import gc
import os
from itertools import product
from textwrap import wrap
import argparse
import abc
import shutil

# typehint imports
if TYPE_CHECKING:
    import transformers as trf
    import torch
    import huggingface_hub
    import pandas as pd
    import numpy as np
    import peft
    import trl
    import matplotlib.pyplot as plt
    import matplotlib
    import sklearn.metrics as skMetr
    import sklearn.model_selection as skMsel
    import tqdm
    import datasets
    LlamaTokenizer = Union[trf.LlamaTokenizer,trf.LlamaTokenizerFast]
    LlamaModel = trf.LlamaForCausalLM

imports = [
    "import transformers as trf",
    "import torch",
    "import huggingface_hub",
    "import pandas as pd",
    "import numpy as np",
    "import peft",
    "import trl",
    "import matplotlib.pyplot as plt",
    "import matplotlib",
    "import sklearn.metrics as skMetr",
    "import sklearn.model_selection as skMsel",
    "import tqdm",
    "import datasets",
]
for i in imports:
    try:
        exec(i)
    except ImportError:
        print(f"Import of {i} failed")
    
try:
    from src.baseline.baseline_functions import * #type: ignore
except Exception:
    pass


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def print_args(func):
    def inner(*args, **kwargs):
        print("Current time:",datetime.datetime.now())
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
    if re.search(pattern_not_severe, answer) is not None or (
        "0" in answer and "1" not in answer
    ):
        return 0
    elif re.search(pattern_severe, answer) is not None or (
        "1" in answer and "0" not in answer
    ):
        return 1
    return -1


def get_tokens(data, tokenizer):
    Ltokens = []
    for i, d in tqdm.tqdm(enumerate(data)):
        gc.collect()
        torch.cuda.empty_cache()
        text = d["text"]
        n_tokens = len(tokenizer.tokenize(text))
        Ltokens.append(n_tokens)
    return Ltokens

def generate_text_with_n_tokens(tokenizer, n: int, base_text: str = "hello") -> List[int]:
    gc.collect()
    torch.cuda.empty_cache()#type: ignore
    [t1,t2] = tokenizer(base_text)["input_ids"]#type:ignore
    if n == 0:
        return [t1]
    else:
        return [t1]+[t2]*(n-1)
    
class MaxTokensEvaluator(abc.ABC):
    @abc.abstractmethod
    def eval(self, n_tokens: int):
        pass

class PipelineMaxTokens(MaxTokensEvaluator):
    def __init__(self, model_name: str, token: str) -> None:
        super().__init__()
        self.tokenizer, model = initialize_model_inference(model_name, token=token) #type: ignore
        self.pipeline = trf.pipeline(
            "text-generation", model=model, tokenizer=self.tokenizer, device_map="auto"
        )
    def eval(self, n_tokens: int):
        token_ids = generate_text_with_n_tokens(self.tokenizer, n_tokens)
        text = self.tokenizer.decode(token_ids[1:]) # 1: to remove the start token
        [answer] = self.pipeline( #type: ignore
                text,
                do_sample=True,
                top_k=1,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
        )
        del answer
        gc.collect()
        torch.cuda.empty_cache()
        
class EmbeddingsMaxTokens(MaxTokensEvaluator):
    def __init__(self, model_name: str, token: str) -> None:
        super().__init__()
        self.tokenizer, self.model = initialize_model_inference(model_name, token=token, hidden_states=True) #type: ignore
    def eval(self, n_tokens: int):
        token_ids = generate_text_with_n_tokens(self.tokenizer, n_tokens)
        embeddings = self.model(torch.tensor([token_ids], dtype=torch.int32)) #type:ignore
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()

class FinetuneMaxTokens(MaxTokensEvaluator):
    def __init__(self, model_name: str, token: str, max_n_tokens: int, base_file_data: Path, n_samples: int = 100, **kwargs_train) -> None:
        super().__init__()
        tokenizer = get_tokenizer(token, model_name)
        self.kwargs = {"model_name": model_name, "token":token, **kwargs_train}
        token_ids = generate_text_with_n_tokens(tokenizer, max_n_tokens)
        token_names = tokenizer.convert_ids_to_tokens(token_ids)
        with open(base_file_data) as f:
            data_preprocessed = json.load(f)
        template = data_preprocessed["template"]
        data_sample = {
            "llama_tokenized_description":token_names,
            "binary_severity": 0
        }
        with open('./tmp.json', 'w') as f:
            json.dump(
            {
                "template": template,
                "data": [data_sample for _ in range(n_samples)]
            }    
            ,f)
    def eval(self, n_tokens: int):
        folder_out = Path("./out_tmp/")
        if folder_out.exists():
            shutil.rmtree(folder_out)
        folder_out.mkdir(parents=True,exist_ok=True)
        main_qlora_generation(
            Path("./tmp.json"),
            folder_out=folder_out,
            field_label="binary_severity",
            field_input="llama_tokenized_description",
            limit_tokens=n_tokens,
            **self.kwargs
        )


def get_max_mix(token_lengths, tokenizer: 'LlamaTokenizer', pipeline: 'trf.Pipeline'):
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
            [answer] = pipeline( #type: ignore
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

def get_tokenizer(token: str, model_name: str) -> 'LlamaTokenizer':
    huggingface_hub.login(token=token)
    tokenizer: 'LlamaTokenizer' = trf.AutoTokenizer.from_pretrained(model_name, use_fast=True,
        cache_dir="/home/$USER/cache_dir")#type: ignore
    return tokenizer
def initialize_model(model_name: str, token: str, hidden_states: bool = False, base_class: Any = trf.AutoModelForCausalLM) -> 'LlamaModel':
    huggingface_hub.login(token=token)
    double_quant_config = trf.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    model = base_class.from_pretrained(
        args.model_name,
        quantization_config=double_quant_config,
        return_dict=hidden_states,
        output_hidden_states=hidden_states,
        cache_dir="/home/$USER/cache_dir"
    )
    return model
def initialize_model_inference(model_name: str, token: str, return_model: bool = True, hidden_states: bool = False) -> Union[Tuple['LlamaTokenizer', 'LlamaModel'],'LlamaTokenizer']:
    huggingface_hub.login(token=token)
    tokenizer = get_tokenizer(token=token, model_name=model_name)
    if return_model:
        model = initialize_model(model_name, token, hidden_states, trf.AutoModelForCausalLM)
        return tokenizer, model
    else:
        return tokenizer

def build_prompt(
    llama_tokenized_template: List[str],
    llama_tokenized_description: List[str],
    template_index_insert: int,
    tokenizer,
    limit_tokens: int,
) -> Tuple[str, List[str]]:
    """We put the template and the description together using the insertion point specified in description_index_insert"""
    ## First make a copy
    tokenized_full_text = llama_tokenized_template[:]
    ## Then remove the <s> token from the description
    description = llama_tokenized_description[1:]
    # limit the number of tokens of the description
    n_tokens_descr_max = limit_tokens - len(tokenized_full_text)
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
    tokenizer, model = initialize_model_inference(model_name,token) #type: ignore
    pipeline = trf.pipeline(
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
        json.dump({"data_path": str(path_data_preprocessed.resolve())}, f, indent=2)
    for i, d in tqdm.tqdm(enumerate(data), total=len(data)):
        # gc.collect()
        # torch.cuda.empty_cache()  # type: ignore
        answer = float("nan")
        severity = float("nan")
        text, tokenized_full_text = build_prompt(
            template["llama_tokenized_template"],
            d["llama_tokenized_description"],
            template["template_index_insert"],
            tokenizer,
            limit_tokens=limit_tokens,
        )
        n_tokens = len(tokenized_full_text)
        assert n_tokens < limit_tokens
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
                folder_predictions / f"predictions_v100l_chunk_{start}_{id_pred}.json",
                "w",
            ) as f:
                json.dump(responses, f)

    with open(
        folder_predictions / f"predictions_v100l_chunk_{start}_{id_pred}.json", "w"
    ) as f:
        json.dump(responses, f, indent=2)


def extract_fields_from_json(folder_path: Path) -> List[Dict]:
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


def compute_metrics(
    folder_predictions: Path,
    path_backup_fields: Path,
    folder_out: Optional[Path] = None,
    input_field: str = "input",
    pred_field: str = "severity_pred",
    mapping_dict: Optional[dict] = None,
    n_tokens_infered_max: int = 7364,
    n_tokens_show_max: int = 7364,
    n_tokens_show_min: int = 0,
    model_name: str = "meta-llama/Llama-2-13b-chat-hf",
    token: str = ""
):
    """Taking the path of the predictions folder, it computes the statistics with the predictions (confusion matrix, precision, recall, f1-score). The confusion matrix is plotted into a png file

    # Arguments
        - folder_predictions: Path, path to the folder where the prediction files are stored
        - folder_out: Path, path to the folder where the statistics will be stored
        - mapping_dict: Optional[Dict], mapping from possible values predicted to name (str or int), default {-2:"Too big query", -1:"Mixed answer", 0: "NON SEVERE", 1: "SEVERE"} and -2 replaces all nan
        - path_backup_fields: path to a file where the missing required fields are put. Must contain bug_id to join with the current data
    # Return
        None
    """
    tokenizer = initialize_model_inference(model_name,token,return_model=False) #type: ignore
    if folder_out is None:
        folder_out = folder_predictions
    fields_data = extract_fields_from_json(folder_predictions)
    data = pd.DataFrame(fields_data)
    # Remove duplicates by bug_id
    data.drop_duplicates(subset="bug_id",inplace=True)
    if 'binary_severity' not in data.columns:
        df_bs = pd.read_json(path_backup_fields)
        # check that we have the same bug_ids in the two dataframes
        assert len(data[data['bug_id'].isin(df_bs)]) == len(data), "Expecting to have all bug_ids of predictions in the backup file"
        data = data.merge(df_bs[['bug_id', 'binary_severity']], on='bug_id', how='left')
    data.to_json(folder_out / "data.json", orient="records", indent=4)
    # Count number of tokens
    data["n_tokens"] = data[input_field].apply(lambda x: len(tokenizer(x)["input_ids"]))  # type: ignore
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
    conf_matrix = skMetr.confusion_matrix(true, pred)

    # Compute F1-score
    f1: 'np.ndarray' = skMetr.f1_score(true, pred, average=None)  # type: ignore
    with open(folder_out / "metrics.json", "w") as f:
        json.dump(
            {
                "date_timestamp": datetime.datetime.now().timestamp(),
                "confusion_matrix": conf_matrix.tolist(),
                "f1": f1.tolist(),
            },
            f,
        )
    possibilities_pred = sorted(
        list(
            set(data["pred"].unique().tolist()).union(
                set(data["true"].unique().tolist())
            )
        )
    )
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
    conf_matrix: 'np.ndarray',
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
        - conf_matrix: 'np.ndarray', confusion matrix
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
    try:
        from pretty_confusion_matrix import pp_matrix
    except Exception:
        from src.llm.llama.pretty_confusion_matrix import pp_matrix
    pp_matrix(  # type: ignor
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
    df: 'pd.DataFrame', path_out: Path, mapping_dict: Dict, n_samples: int = 5
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

def convert_dict_to_str(data):
    if isinstance(data, dict):
        return {key: convert_dict_to_str(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_dict_to_str(item) for item in data]
    elif isinstance(data, int) or isinstance(data, float) or isinstance(data, str):
        return data
    else:
        return str(data)
class DataInputTrain(TypedDict):
    bug_id: int
    binary_severity: int
    description: str
    llama_tokenized_description: List[str]
class TemplateDict(TypedDict):
    template: str
    llama_tokenized_template: List[str]
    template_index_insert: int
    
class AbstractTask(abc.ABC):
    def __init__(self, lora_alpha: int, lora_dropout: float, r: int):
        self.r = r
        self.lora_dropout = lora_dropout
        self.lora_alpha = lora_alpha
    @abc.abstractmethod
    def get_lora_config(self) -> 'peft.LoraConfig':
        pass
    @abc.abstractmethod
    def get_dict_model(self, d: DataInputTrain, input: List[str], output: str, template: TemplateDict, tokenizer: 'LlamaTokenizer', limit_tokens: int) -> dict:
        pass
    @abc.abstractmethod
    def get_model(self, model_name: str, token: str) -> Any:
        pass
    @abc.abstractmethod
    def train(self):
        pass
class TextGenerationTask(AbstractTask):
    def get_lora_config(self) -> 'peft.LoraConfig':
        return peft.LoraConfig(  # type: ignore
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.r,
            bias="none",
            inference_mode=False,   
            task_type="CAUSAL_LM",
        )
    def get_dict_model(self, d: DataInputTrain, input: List[str], output: str, template: TemplateDict, tokenizer: 'LlamaTokenizer', limit_tokens: int) -> dict:
        template["llama_tokenized_template"] = template[
                "llama_tokenized_template"
            ] + ["<0x0A>", str(d["binary_severity"])]
        text, tokenized_full_text = build_prompt(
            template["llama_tokenized_template"],
            input,
            template["template_index_insert"],
            tokenizer,
            limit_tokens=limit_tokens,
        )
        return {**d, "text": text, "tokenized_full_text": tokenized_full_text}
    def get_model(self, model_name: str, token: str):
        return initialize_model(model_name=model_name, token=token)
    def train(self):
        trainer = trl.SFTTrainer(  # type: ignore
            model=model,
            train_dataset=train_dataset,  #type: ignore
            eval_dataset=valid_dataset,  #type: ignore
            peft_config=peft_config,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False,
            max_seq_length=limit_tokens
        )
class TextClassificationTask(AbstractTask):
    def get_lora_config(self) -> 'peft.LoraConfig':
        return peft.LoraConfig(# type: ignore
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.r,
            bias="none",
            inference_mode=False,   
            task_type="SEQ_CLS", 
        )
    def get_dict_model(self, d: DataInputTrain, input: List[str], output: str, template: TemplateDict, tokenizer: 'LlamaTokenizer', limit_tokens: int) -> dict:
        template["llama_tokenized_template"] = template[
                "llama_tokenized_template"
            ] + ["<0x0A>", str(d["binary_severity"])]
        text, tokenized_full_text = build_prompt(
            template["llama_tokenized_template"],
            input,
            template["template_index_insert"],
            tokenizer,
            limit_tokens=limit_tokens,
        )
        return {**d, "text": text, "tokenized_full_text": tokenized_full_text}
    def get_model(self, model_name: str, token: str) -> Any:
        model = initialize_model(model_name=model_name, token=token, base_class=trf.AutoModelForSequenceClassification)
        config = self.get_lora_config()
        return peft.get_peft_model(model, config)
class Evaluator:
    def __init__(self, *names: str) -> None:
        self.names: Tuple[str, ...] = names
        self.reset()
    def reset(self):
        self.buffer = {k:[] for k in self.names}
    def add_samples(self, **kwargs):
        for k,v in kwargs.items():
            self.buffer[k].append(v)
        
@print_args
def main_qlora_classification(
    file_examples: Path,
    folder_out: Path,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_r: int = 64,
    model_name: str = "meta-llama/Llama-2-13b-chat-hf",
    token: str = "",
    field_label: str = "binary_severity",
    field_input: str = "llama_tokenized_description",
    num_train_epochs: int = 1,
    tr_bs: int = 1,
    val_bs: int = 1,
    learning_rate: float = 2e-4,
    train_size: float = 0.3,
    limit_tokens: int = 7364,
    new_model_name: str = "",
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
        - lora_alpha: int, scaling factor for the weight matrices. alpha is a scaling factor that adjusts the magnitude of the combined result (base model output + low-rank adaptation). Default: 16
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
        huggingface_hub.login(token=token)
    if not folder_out.exists():
        folder_out.mkdir(parents=True)

    peft_config = peft.LoraConfig(  # type: ignore
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        inference_mode=False,   
        task_type="CAUSAL_LM",
    )
    tokenizer: LlamaTokenizer = initialize_model_inference(model_name, token, return_model=False) #type: ignore
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = initialize_model(model_name=model_name, token=token, base_class=trf.AutoModelForSequenceClassification)
    model = peft.get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.print_trainable_parameters()
    # create datasets
    print("Create datasets")
    with open(file_examples) as f:
        data_preprocessed = json.load(f)
    data = data_preprocessed["data"]
    template = data_preprocessed["template"]
    L = []
    for d in data:
        template["llama_tokenized_template"] = template[
                "llama_tokenized_template"
            ] + ["<0x0A>", str(d["binary_severity"])]
        text, tokenized_full_text = build_prompt(
            template["llama_tokenized_template"],
            d[field_input],
            template["template_index_insert"],
            tokenizer,
            limit_tokens=limit_tokens,
        )
        d = {**d, "text": text, "tokenized_full_text": tokenized_full_text}
        L.append(d)
    idx_tr, idx_val = skMsel.train_test_split(np.arange(len(L)), train_size=train_size)  # type: ignore
    idx_tr = set(idx_tr)
    idx_val = set(idx_val)
    print("Building dataloaders")
    def collate_fn(data: List[dict]):
        inputs = tokenizer([d['text'] for d in data])
        outputs = [d[field_label] for d in data]
        yield inputs,outputs
    tr_data,val_data = [],[]
    for i,e in enumerate(L):
        if i in idx_tr:
            tr_data.append(e)
        if i in idx_val:
            val_data.append(e)
    train_dataloader = torch.utils.data.DataLoader(tr_data, shuffle=True, collate_fn=collate_fn, batch_size=tr_bs)
    eval_dataloader = torch.utils.data.DataLoader(val_data, shuffle=False, collate_fn=collate_fn, batch_size=val_bs)
    print("Starting training QLORA")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    # Instantiate scheduler
    lr_scheduler = trf.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * num_train_epochs),
        num_training_steps=(len(train_dataloader) * num_train_epochs),
    )
    device = "cuda"
    model.to(device)
    for epoch in range(num_train_epochs):
        model.train()
        for step, (inputs, labels) in enumerate(tqdm.tqdm(train_dataloader)):
            inputs.to(device)
            outputs = model(inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            break

        model.eval()
        
        for step, (inputs, labels) in enumerate(tqdm.tqdm(eval_dataloader)):
            inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            print(f"{outputs=},{outputs.shape=}")
            predictions = outputs.logits.argmax(dim=-1)
            break
        break
        # eval_metric = metric.compute()
        # print(f"epoch {epoch}:", eval_metric)
    print("Saving trained QLORA model")
    # if new_model_name != "":
    #     output_dir = "./trained_model"
    #     trainer.model.save_pretrained(output_dir)
    #     trainer.model.push_to_hub(new_model_name)

@print_args
def main_qlora_generation(
    file_examples: Path,
    folder_out: Path,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_r: int = 64,
    model_name: str = "meta-llama/Llama-2-13b-chat-hf",
    token: str = "",
    field_label: str = "binary_severity",
    field_input: str = "llama_tokenized_description",
    num_train_epochs: int = 1,
    tr_bs: int = 1,
    val_bs: int = 1,
    optim: str = "paged_adamw_32bit",
    save_steps: int = 25,
    logging_steps: int = 25,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.001,
    fp16: bool = False,
    bf16: bool = False,
    max_grad_norm: float = 0.3,
    max_steps: int = -1,
    warmup_ratio: float = 0.03,
    group_by_length: bool = True,
    lr_scheduler_type: str = "constant",
    eval_steps: int = 20,
    train_size: float = 0.3,
    limit_tokens: int = 7364,
    new_model_name: str = "",
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
        - lora_alpha: int, scaling factor for the weight matrices. alpha is a scaling factor that adjusts the magnitude of the combined result (base model output + low-rank adaptation). Default: 16
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
        huggingface_hub.login(token=token)
    if not folder_out.exists():
        folder_out.mkdir(parents=True)

    tokenizer,model = initialize_model_inference(model_name, token, return_model=True) #type: ignore
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    peft_config = peft.LoraConfig(# type: ignore
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            inference_mode=False,   
            task_type="SEQ_CLS", 
        )
    # Set training parameters
    training_arguments = trf.TrainingArguments(
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
        report_to="all",  # type: ignore
        evaluation_strategy="steps",
        eval_steps=eval_steps
    )
    # create datasets
    print("Create datasets")
    train_path = folder_out / f"train_max_{limit_tokens}.json"
    valid_path = folder_out / f"valid_max_{limit_tokens}.json"
    full_path = folder_out / f"full_max_{limit_tokens}.json"
    if not train_path.exists() or not valid_path.exists():
        with open(file_examples) as f:
            data_preprocessed = json.load(f)
        data = data_preprocessed["data"]
        template = data_preprocessed["template"]
        L = []
        for d in data:
            template["llama_tokenized_template"] = template[
                    "llama_tokenized_template"
                ] + ["<0x0A>", str(d[field_label])]
            text, tokenized_full_text = build_prompt(
                template["llama_tokenized_template"],
                d[field_input],
                template["template_index_insert"],
                tokenizer,
                limit_tokens=limit_tokens,
            )
            d = {**d, "text": text, "tokenized_full_text": tokenized_full_text}
            L.append(d)
        idx_tr, idx_val = skMsel.train_test_split(np.arange(len(L)), train_size=train_size)  # type: ignore
        idx_tr = set(idx_tr)
        idx_val = set(idx_val)
        with open(full_path, "w") as f:
            json.dump(L, f)
        with open(train_path, "w") as f:
            json.dump([e for i, e in enumerate(L) if i in idx_tr], f)
        with open(valid_path, "w") as f:
            json.dump([e for i, e in enumerate(L) if i in idx_val], f)
    train_dataset = datasets.load_dataset("json", data_files=str(train_path.resolve()), split="train")  # type: ignore
    valid_dataset = datasets.load_dataset("json", data_files=str(valid_path.resolve()), split="train")  # type: ignore
    # Set supervised fine-tuning parameters
    print("Set supervised fine-tuning parameters")
    trainer = trl.SFTTrainer(  # type: ignore
        model=model,
        train_dataset=train_dataset,  #type: ignore
        eval_dataset=valid_dataset,  #type: ignore
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        max_seq_length=limit_tokens
    )
    print("Starting training QLORA")
    trainer.train()
    print("Saving trained QLORA model")
    if new_model_name != "":
        output_dir = "./trained_model"
        trainer.model.save_pretrained(output_dir)
        trainer.model.push_to_hub(new_model_name)
def get_max_tokens(evaluator: MaxTokensEvaluator, min_token_length: int = 0, max_token_length: int = 5000):
    """
    The main algorithm comes from get_max_mix. 
    However dependency injection allows to provide the specific element to evaluate with MaxTokensEvaluator
    """
    
    min_token_length = max(1,min_token_length) # because start token
    max_work = 0
    min_not_work = float("inf")

    while min_token_length < max_token_length:
        mid_token_length = (min_token_length + max_token_length) // 2
        try:
            evaluator.eval(mid_token_length)
            # If the code above works, update max_work and adjust the search range
            max_work = mid_token_length
            min_token_length = mid_token_length + 1
        except Exception as e:
            print(e)
            # If the code above raises an exception, update min_not_work and adjust the search range
            min_not_work = mid_token_length
            max_token_length = mid_token_length - 1
        try:
            gc.collect()
            torch.cuda.empty_cache()#type: ignore
        except Exception as e:
            print("Exception clear")
            print(e)
            print("End exception clear")
    return (max_work, min_not_work)

@print_args
def get_llama2_embeddings(
    model_name: str,
    path_data_preprocessed: Path,
    layers_ids: Optional[Tuple[int]] = None,
    start: int = 0,
    end: int = -1,
    limit_tokens: int = 7364,
    id_pred: str = "",
):
    if layers_ids is None:
        layers_ids = (0,)
    tokenizer, model = initialize_model_inference(model_name, token, hidden_states=True)#type: ignore
    with open(path_data_preprocessed) as f:
        data_preprocessed = json.load(f)
    data = data_preprocessed["data"]
    if end == -1:
        end = len(data)
    data = data[start:end]
    print(f"Running for {start=} {end=}")
    folder_predictions = path_data_preprocessed.parent / "embeddings"
    folder_predictions.mkdir(exist_ok=True, parents=True)
    
    for idx_layer in layers_ids:
        with open(
            folder_predictions / f"embeddings_chunk_{id_pred}_layer_{idx_layer}_{start}.json",
            "w",
        ) as f:
            f.write("")
    for i, d in tqdm.tqdm(enumerate(data), total=len(data)):
        text = d["description"]
        tokenized_full_text = tokenizer.encode(text)
        tokenized_full_text = tokenized_full_text[:limit_tokens]
        embeddings = model(torch.tensor([tokenized_full_text], dtype=torch.int32))  # type: ignore
        for idx_layer in layers_ids:
            with open(
                folder_predictions / f"embeddings_chunk_{id_pred}_layer_{idx_layer}_{start}.json",
                "a",
            ) as f:
                f.write(str({
                    **d,
                    "layer_id":idx_layer,
                    "hidden_state":embeddings.hidden_states[idx_layer].tolist()[0],
                    "text":text,
                    "tokenized": tokenizer.convert_ids_to_tokens(tokenized_full_text)
                    })+",\n")
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()  # type: ignore


class EmbeddingDict(TypedDict, total=False):
    """Contains especially
    - description: str, the field that has been used to generate the embeddings. Could have been truncated
    - layer_id: int, the id of the layer that have been taken into hidden_representation
    - hidden_state: List, to conver to array or torch Tensor, the actual hidden representation of shape (seq_length, vocab_size)
    - text: str, same field as description, the text that has been sent to llama2 before tokenization and limiting the number of tokens
    - tokenized: List[int], the list of tokens ids after llama2 tokenizer and truncation
    - bug_id: int, the id of the bug
    """
    description: str
    layer_id: int
    hidden_state: List[List[float]]
    text: str
    tokenized: List[int]
    bug_id: int


    
def get_data_embeddings(folder_embeddings: Path, layer_id: int = -1, base_name: str = "embeddings_chunk__trunc_") -> Generator[EmbeddingDict, None, None]:
    sorted_path = list(folder_embeddings.rglob(f"{base_name}layer_{layer_id}_*.json"))
    sorted_path = sorted(sorted_path,key=lambda x:int(x.name.split(".")[0].split("_")[-1]))
    for p in sorted_path:
        print("Reading ",p)
        with open(p, 'r') as json_file:
            for i,(line) in enumerate(json_file):
                # Load and process each line as JSON data.
                try:
                    data = eval(line[:-2]) #-2 to remove the coma and the back to line
                    yield data
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    

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
        "max_tokens_pipeline",
        "max_tokens_embeddings",
        "max_tokens_finetuning",
        "finetune_generation",
        "finetune_classification",
        "finetune",
        "embeddings_gen",
        "nn_embedding",
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
        type=str,
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
        "-pred_field",
        type=str,
        help="Predicted field to use",
        default="severity_pred_trunc",
    )
    parser.add_argument(
        "-input_field", type=str, help="Predicted field to use", default="description"
    )
    parser.add_argument(
        "-n_tokens_infered_max",
        type=int,
        help="Number of maximum infered tokens",
        default=7364,
    )
    parser.add_argument(
        "-n_tokens_show_min",
        type=int,
        help="Maximum number of tokens tokens in the statistics",
        default=0,
    )
    parser.add_argument(
        "-n_tokens_show_max",
        type=int,
        help="Minimum number of tokens shown in the statistics",
        default=7364,
    )
    parser.add_argument(
        "-n_data", type=int, help="Total number of data in the dataset", default=22302
    )
    parser.add_argument(
        "-id", type=str, help="Id to put on the files to save", default="_trunc"
    )
    parser.add_argument(
        "-model_name",
        type=str,
        help="Name of the huggingface model to use",
        default="meta-llama/Llama-2-13b-chat-hf",
    )
    parser.add_argument(
        "-new_model_name",
        type=str,
        help="Name of the huggingface model to use",
        default="meta-llama/Finetune-Llama-2-13b-chat-hf",
    )
    parser.add_argument(
        "-qlora_alpha",
        type=float,
        help="Ponderation of the QLORA finetuning",
        default=8,
    )
    parser.add_argument(
        "-qlora_dropout",
        type=float,
        help="Dropout applied to the QLORA finetuning",
        default=0.1,
    )
    parser.add_argument(
        "-qlora_r",
        type=int,
        help="Rank of the matrices of the QLORA finetuning",
        default=8,
    )
    parser.add_argument(
        "-num_train_epochs",
        type=int,
        help="Number of training epochs for qlora",
        default=5,
    )
    parser.add_argument(
        "-layers_ids",
        type=str,
        help="Layers ids for the embedding to take",
        default="(0,)",
    )
    parser.add_argument(
        "-base_name",
        type=str,
        help="Base name of the json file with the layer id for get_data_embeddings (ex: embeddings_chunk__trunc_layer_-1_4460.json will give embeddings_chunk__trunc_)",
        default="(0,)",
    )
    parser.add_argument(
        "-path_backup_fields",
        type=str,
        help="Allow to add the missing field of binary_severity based on the bug_id common field. Relative path to the data path",
        default="llm/data_preprocessed_tokens.json",
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
        intervals[-1][1] = n_data
        [seed_start, seed_end] = intervals[args.interval_idx]
    else:
        [seed_start, seed_end] = [args.seed_start, args.seed_end]

    if args.algorithm == "inference":
        main_inference(
            args.path_data_json,
            token=args.token,
            start=seed_start,
            end=seed_end,
            model_name=args.model_name,
            id_pred=args.id,
            limit_tokens=args.n_tokens_infered_max,
        )
    elif args.algorithm == "compute_metrics":
        path_out = args.path_data_folder / f"out_{args.pred_field}{args.id}"
        path_out.mkdir(parents=True, exist_ok=True)
        compute_metrics(
            args.path_data_folder,
            args.path_data_folder / args.path_backup_fields,
            path_out,
            pred_field=args.pred_field,
            input_field=args.input_field,
            token=args.token,
            n_tokens_infered_max=args.n_tokens_infered_max,
            n_tokens_show_min=args.n_tokens_show_min,
            n_tokens_show_max=args.n_tokens_show_max,
        )
    elif args.algorithm == "finetune_generation":
        path_out = args.path_data_folder / f"qlora_finetune_generation_{args.id}"
        path_out.mkdir(parents=True, exist_ok=True)
        main_qlora_generation(
            model_name=args.model_name,
            file_examples=args.path_data_json,
            new_model_name=args.new_model_name,
            folder_out=path_out,
            token=args.token,
            field_label="binary_severity",
            field_input=args.input_field,
            lora_alpha=args.qlora_alpha,
            lora_dropout=args.qlora_dropout,
            lora_r=args.qlora_r,
            limit_tokens=args.n_tokens_infered_max,
            num_train_epochs=args.num_train_epochs
        )
    elif args.algorithm == "finetune_classification":
        path_out = args.path_data_folder / f"qlora_finetune_classification_{args.id}"
        path_out.mkdir(parents=True, exist_ok=True)
        main_qlora_classification(
            model_name=args.model_name,
            file_examples=args.path_data_json,
            new_model_name=args.new_model_name,
            folder_out=path_out,
            token=args.token,
            field_label="binary_severity",
            field_input=args.input_field,
            lora_alpha=args.qlora_alpha,
            lora_dropout=args.qlora_dropout,
            lora_r=args.qlora_r,
            limit_tokens=args.n_tokens_infered_max,
            num_train_epochs=args.num_train_epochs
        )
    elif args.algorithm == "max_tokens_pipeline":
        (max_work, min_not_work) = get_max_tokens(
            PipelineMaxTokens(
                model_name=args.model_name,
                token=args.token
            ),
            min_token_length=1,
            max_token_length=10000
        )
        print(f"{max_work=},{min_not_work=}")
    elif args.algorithm == "max_tokens_embeddings":
        (max_work, min_not_work) = get_max_tokens(
            EmbeddingsMaxTokens(
                model_name=args.model_name,
                token=args.token
            ),
            min_token_length=1,
            max_token_length=10000
        )
        print(f"{max_work=},{min_not_work=}")
    elif args.algorithm == "max_tokens_finetuning":
        path_data_folder = Path(args.path_data_folder)
        if not path_data_folder.exists():
            raise ValueError(f"The path {path_data_folder} does not exist")
        max_n_tokens = 10000
        args_dict = convert_dict_to_str(vars(args))
        L = []
        for model_name in ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"]:
            for qlora_r in [2,8,16,32,64,128,256]:
                (max_work, min_not_work) = get_max_tokens(
                    FinetuneMaxTokens(
                        model_name=model_name,
                        token=args.token,
                        max_n_tokens=max_n_tokens,
                        base_file_data=args.path_data_json,
                        lora_alpha=args.qlora_alpha,
                        lora_dropout=args.qlora_dropout,
                        lora_r=qlora_r,
                    ),
                    min_token_length=1,
                    max_token_length=max_n_tokens
                )
                L.append({"max_work":max_work, "min_not_work": min_not_work, "qlora_r":qlora_r, "model_name": args.model_name, "args":args_dict})
                with open(path_data_folder / "finetune_tokens_lim.json", "w") as f:
                    json.dump(L,f)
    elif args.algorithm == "embeddings_gen":
        layers_ids = eval(args.layers_ids)
        get_llama2_embeddings(
            model_name=args.model_name,
            path_data_preprocessed=args.path_data_json,
            layers_ids=layers_ids,
            start=seed_start,
            end=seed_end,
            id_pred=args.id,
            limit_tokens=args.n_tokens_infered_max,
        )
    elif args.algorithm == "nn_embedding":
        folder_embeddings = Path(args.path_data_folder) / "embeddings"
        layer_id: Tuple[int] = eval(args.layers_ids)
        if len(layer_id) > 1:
            raise ValueError(f"Expecting just one layer id not {len(layer_id)}")
        print(args.algorithm)

        for d in get_data_embeddings(
            folder_embeddings=folder_embeddings,
            layer_id=layer_id[0],
            base_name=args.base_name,
        ):
            # with open("/home/rmoine/tmp.txt","w") as f:
            #     f.write(str(d))
            line_dict = json.loads(d)
            bug_id = line_dict["bug_id"]
            binary_severity = line_dict["binary_severity"]
            hidden_state = np.array(line_dict["hidden_state"])

            base_name=args.base_name
            came_from = list(folder_embeddings.rglob(f"{base_name}layer_{layer_id[0]}_*.json"))

            sum_aggregated_array = np.sum(hidden_state, axis=0)
            mean_aggregated_array = sum_aggregated_array / len(hidden_state)

            aggregated_list = mean_aggregated_array.tolist()
            
            data = {
                "bug_id": bug_id,
                "binary_severity": binary_severity,
                "from": came_from,
                "aggregated_list": aggregated_list,
            }
            with open(Path(args.path_data_folder) / 'output_file.json', 'a') as outfile:
                json.dump(data, outfile)
                outfile.write('\n')