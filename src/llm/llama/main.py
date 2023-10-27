from pathlib import Path
from typing import *  # type: ignore
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
import logging
import subprocess
import psutil
import multiprocessing as mp
import functools

# typehint imports
if TYPE_CHECKING:
    import transformers as trf
    import torch
    import torch.nn as nn
    import torch.utils.data as dt
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
    import h5py

    LlamaTokenizer = Union[trf.LlamaTokenizer, trf.LlamaTokenizerFast]
    LlamaModel = trf.LlamaForCausalLM
    PoolingOperationCode = Literal["mean","sum"]
    PoolingFn = Callable[[torch.Tensor],torch.Tensor]
    ModelName = Literal["meta-llama/Llama-2-13b-chat-hf","meta-llama/Llama-2-7b-chat-hf","meta-llama/Llama-2-70b-chat-hf"]

imports = [
    "import transformers as trf",
    "import torch",
    "import torch.nn as nn",
    "import torch.utils.data as dt",
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
    "import h5py",
]
for i in imports:
    try:
        exec(i)
    except ImportError:
        print(f"Import of {i} failed")

try:
    from src.baseline.baseline_functions import *  # type: ignore
except Exception:
    pass

try:
    from src.llm.llama.class_nn import SimpleNN
except Exception:
    pass

class PreprocessedData(TypedDict, total=False):
    """
    - bug_id: int, the id of the bug
    - binary_severity: int, the severity level of the bug 0=NON SEVERE, 1=SEVERE
    - description: str, the field that has been used to generate the embeddings
    - stemmed_description: str, the field that has been used to generate the embeddings
    """
    bug_id: int
    binary_severity: int
    description: str
    stemmed_description: str
    

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


class CustomFormatter(logging.Formatter):
    def __init__(
        self, fmt=None, datefmt=None, style="%", validate: bool = True
    ) -> None:
        super().__init__(fmt, datefmt, style, validate)
        try:
            self.total_ram_gpu = float(
                subprocess.check_output("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0",shell=True,)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            self.total_ram_gpu = None
            pass

    def format(self, record):
        # Log CPU RAM usage
        ram = psutil.virtual_memory()
        cpu_ram_message = f"RAM {ram.used / (1024 ** 3):.3f}/{ram.total / (1024 ** 3):.3f}GB ({ram.used/ram.total:.2f}%)"

        # Log GPU VRAM usage (assuming a single GPU, adjust as needed)
        if self.total_ram_gpu is not None:
            used = float(
                subprocess.check_output(
                    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0",
                    shell=True,
                )
                .decode("utf-8")
                .strip()
            )
            gpu_vram_message = f"GPU VRAM {used / (1024 ** 3):.3f}/{self.total_ram_gpu / (1024 ** 3):.3f}GB ({used/self.total_ram_gpu:.2f}%)"
        else:
            gpu_vram_message = "GPU VRAM <nan>"

        # Add the CPU RAM and GPU VRAM messages to the log record
        record.cpu_ram_message = cpu_ram_message
        record.gpu_vram_message = gpu_vram_message

        return super().format(record)
    
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
custom_formatter = CustomFormatter(
    "%(asctime)s - %(levelname)s - %(message)s - %(cpu_ram_message)s - %(gpu_vram_message)s"
)
handler.setFormatter(custom_formatter)
logger.addHandler(handler)


def print_args(func):
    def inner(*args, **kwargs):
        print("Current time:", datetime.datetime.now())
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


def generate_text_with_n_tokens(
    tokenizer, n: int, base_text: str = "hello"
) -> List[int]:
    gc.collect()
    torch.cuda.empty_cache()  # type: ignore
    [t1, t2] = tokenizer(base_text)["input_ids"]  # type:ignore
    if n == 0:
        return [t1]
    else:
        return [t1] + [t2] * (n - 1)


class MaxTokensEvaluator(abc.ABC):
    @abc.abstractmethod
    def eval(self, n_tokens: int):
        pass


class PipelineMaxTokens(MaxTokensEvaluator):
    def __init__(self, model_name: str, token: str) -> None:
        super().__init__()
        self.tokenizer, model = initialize_model_inference(model_name, token=token)  # type: ignore
        self.pipeline = trf.pipeline(
            "text-generation", model=model, tokenizer=self.tokenizer, device_map="auto"
        )

    def eval(self, n_tokens: int):
        token_ids = generate_text_with_n_tokens(self.tokenizer, n_tokens)
        text = self.tokenizer.decode(token_ids[1:])  # 1: to remove the start token
        [answer] = self.pipeline(  # type: ignore
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
        self.tokenizer, self.model = initialize_model_inference(model_name, token=token, hidden_states=True)  # type: ignore

    def eval(self, n_tokens: int):
        token_ids = generate_text_with_n_tokens(self.tokenizer, n_tokens)
        embeddings = self.model(
            torch.tensor([token_ids], dtype=torch.int32)
        )  # type:ignore
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()


class FinetuneMaxTokens(MaxTokensEvaluator):
    def __init__(
        self,
        model_name: str,
        token: str,
        max_n_tokens: int,
        base_file_data: Path,
        n_samples: int = 100,
        **kwargs_train,
    ) -> None:
        super().__init__()
        tokenizer = get_tokenizer(token, model_name)
        self.kwargs = {"model_name": model_name, "token": token, **kwargs_train}
        token_ids = generate_text_with_n_tokens(tokenizer, max_n_tokens)
        token_names = tokenizer.convert_ids_to_tokens(token_ids)
        with open(base_file_data) as f:
            data_preprocessed = json.load(f)
        template = data_preprocessed["template"]
        data_sample = {"llama_tokenized_description": token_names, "binary_severity": 0}
        with open("./tmp.json", "w") as f:
            json.dump(
                {"template": template, "data": [data_sample for _ in range(n_samples)]},
                f,
            )

    def eval(self, n_tokens: int):
        folder_out = Path("./out_tmp/")
        if folder_out.exists():
            shutil.rmtree(folder_out)
        folder_out.mkdir(parents=True, exist_ok=True)
        main_qlora_generation(
            Path("./tmp.json"),
            folder_out=folder_out,
            field_label="binary_severity",
            field_input="llama_tokenized_description",
            limit_tokens=n_tokens,
            **self.kwargs,
        )


def get_max_mix(token_lengths, tokenizer: "LlamaTokenizer", pipeline: "trf.Pipeline"):
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
            [answer] = pipeline(  # type: ignore
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


def get_tokenizer(token: str, model_name: str) -> "LlamaTokenizer":
    huggingface_hub.login(token=token)
    tokenizer: "LlamaTokenizer" = trf.AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir="/project/def-aloise/rmoine/cache_dir",
        token=token,
    )  # type: ignore
    return tokenizer


def initialize_model(
    model_name: str,
    token: str,
    hidden_states: bool = False,
    base_class: Any = trf.AutoModelForCausalLM,
) -> "LlamaModel":
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
        cache_dir="/project/def-aloise/rmoine/cache_dir",
        token=token,
    )
    return model


def initialize_model_inference(
    model_name: str, token: str, return_model: bool = True, hidden_states: bool = False
) -> Union[Tuple["LlamaTokenizer", "LlamaModel"], "LlamaTokenizer"]:
    huggingface_hub.login(token=token)
    tokenizer = get_tokenizer(token=token, model_name=model_name)
    if return_model:
        model = initialize_model(
            model_name, token, hidden_states, trf.AutoModelForCausalLM
        )
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
    tokenizer, model = initialize_model_inference(model_name, token)  # type: ignore
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
                f"severity_pred": severity,
                f"input": text,
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


def compute_metrics_from_list(
    fields_data: List,
    tokenizer: Optional[Any] = None,
    path_backup_fields: Optional[Path] = None,
    input_field: str = "input",
    pred_field: str = "severity_pred",
    n_tokens_infered_max: int = 7364,
    n_tokens_show_max: int = 7364,
    n_tokens_show_min: int = 0,
    mapping_dict: Optional[dict] = None
):
    data = pd.DataFrame(fields_data)
    # Remove duplicates by bug_id
    data.drop_duplicates(subset="bug_id", inplace=True)
    if "binary_severity" not in data.columns:
        if  path_backup_fields is not None:
            df_bs = pd.read_json(path_backup_fields)
            # check that we have the same bug_ids in the two dataframes
            assert len(data[data["bug_id"].isin(df_bs)]) == len(
                data
            ), "Expecting to have all bug_ids of predictions in the backup file"
            data = data.merge(df_bs[["bug_id", "binary_severity"]], on="bug_id", how="left")
        else:
            raise Exception("Missing field binary_severity")
    # Count number of tokens
    if tokenizer is not None:
        data["n_tokens"] = data[input_field].apply(lambda x: len(tokenizer(x)["input_ids"]))  # type: ignore
    data_full = data.copy()
    # Filter by limit of tokens
    if tokenizer is not None:
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
    f1: List[float] = skMetr.f1_score(true, pred, average=None).tolist()  # type: ignore
    return conf_matrix, f1, data_full

def compute_metrics_from_files(
    conf_matrix: "np.ndarray", 
    f1: List[float], 
    folder_out: Path,
    data_full: Optional[pd.DataFrame] = None,
    true_field: str = "binary_severity",
    pred_field: str = "severity_pred",
    mapping_dict: Optional[dict] = None,
    n_tokens_infered_max: int = 7364,
    n_tokens_show_max: int = 7364,
    n_tokens_show_min: int = 0,
    backend: Optional[str] = "agg"
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
    if data_full is not None:
        data_full.to_json(folder_out / "data.json", orient="records", indent=4)
    with open(folder_out / "metrics.json", "w") as f:
        json.dump(
            {
                "date_timestamp": datetime.datetime.now().timestamp(),
                "confusion_matrix": conf_matrix.tolist(),
                "f1": f1,
            },
            f,
        )
    if data_full is not None:
        possibilities_pred = sorted(
            list(
                set(data[true_field].unique().tolist()).union(
                    set(data[pred_field].unique().tolist())
                )
            )
        )
    else:
        possibilities_pred = list(range(len(conf_matrix)))
    if mapping_dict is None:
        mapping_dict = {
            -2: f"Too big >={n_tokens_infered_max}",
            -1: "Mixed answer",
            0: "NON SEVERE",
            1: "SEVERE",
        }
    id = f"_field_{pred_field}_shown_{n_tokens_show_min}_{n_tokens_show_max}_trunc_{n_tokens_infered_max}"
    plot_confusion(
        conf_matrix=conf_matrix,
        folder_path=folder_out,
        mapping_dict=mapping_dict,
        unique_values=possibilities_pred,
        limit_tokens=n_tokens_infered_max,
        backend=backend,  # type: ignore
        title=f"Confusion matrix\nfor field {pred_field}\n{n_tokens_infered_max=}\nn_tokens_shown in [{n_tokens_show_min};{n_tokens_show_max}[",
        id=id,
    )
    if data_full is not None:
        # find representants
        find_representant(
            data_full,
            folder_out / f"representants{id}.json",
            n_samples=5,
            mapping_dict=mapping_dict,
        )


def plot_confusion(
    conf_matrix: "np.ndarray",
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
        try:
            from src.llm.llama.pretty_confusion_matrix import pp_matrix
        except Exception:
            from llama.pretty_confusion_matrix import pp_matrix
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



class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj.resolve())  # Serialize the path as a string
        return super().default(obj)

def find_representant(
    df: "pd.DataFrame", path_out: Path, mapping_dict: Dict, n_samples: int = 5
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


class Evaluator:
    def __init__(self) -> None:
        self.reset()
    def reset(self):
        self.buffer = []

    def add_samples(self, d: Dict, pred: List[float], logits: List[float]):
        for d_e,pred_e,logits_e in zip(d, pred, logits):
            d_saved = {**d_e, "pred": pred_e, "logits": logits_e}
            self.buffer.append(d_saved)
    
    def get_data(self):
        return self.buffer


@print_args
def generate_dataset(
    folder_out: Path,
    file_examples: Path,
    file_split: Path,
    field_input: str,
    token: str,
    model_name: str,
    limit_tokens: int,
    id: str = "",
):
    """Generates the dataset for the finetuning"""
    logger.info("Create datasets")
    id = id.replace("/","_")
    train_path = folder_out / f"finetune_train{id}.json"
    valid_path = folder_out / f"finetune_valid{id}.json"
    full_path = folder_out / f"finetune_full{id}.json"
    if not train_path.exists() or not valid_path.exists():
        tokenizer = get_tokenizer(token,model_name)
        logger.info("-> Creating cache")
        with open(file_examples) as f:
            data_preprocessed = json.load(f)
        data: List[PreprocessedData] = data_preprocessed["data"]
        template = data_preprocessed["template"]
        with open(file_split) as f:
            splits = {k:set(v) for k,v in json.load(f).items()}
        L = {"tr":[],"val":[],"test":[]}

        for d in tqdm.tqdm(data):
            text, tokenized_full_text = build_prompt(
                template["llama_tokenized_template"],
                d[field_input],
                template["template_index_insert"],
                tokenizer,
                limit_tokens=limit_tokens,
            )
            d = {**d, "input": text, "n_tokens": len(tokenized_full_text)}
            for k in ["tr","val","test"]:
                if d['bug_id'] in splits[k]:
                    L[k].append(d)
                    break
        logger.info("-> End creation cache in memory")
        with open(full_path, "w") as f:
            json.dump(L, f)
        with open(train_path, "w") as f:
            json.dump(L['tr'], f)
        with open(valid_path, "w") as f:
            json.dump(L['val'], f)
        return L['tr'], L['val'], train_path, valid_path
    else:
        logger.info("-> Reading cache")
        with open(train_path, "r") as f:
            train_data = json.load(f)
        with open(valid_path, "r") as f:
            valid_data = json.load(f)
        return train_data, valid_data, train_path, valid_path


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
    mapping_dict: Optional[dict] = None,
    lim_size: int = 500
) -> Tuple['np.ndarray',List[float],'pd.DataFrame']:
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
        - lim_size: int = -1, the size of the training and validation set. If -1 no limit
    """
    print("main_qlora_classification")
    logger.info("main_qlora")
    if token != "":
        huggingface_hub.login(token=token)
    if not folder_out.exists():
        folder_out.mkdir(parents=True)

    logger.info("peft.LoraConfig")
    peft_config = peft.LoraConfig(  # type: ignore
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        inference_mode=False,
        task_type="SEQ_CLS",
    )
    logger.info("LlamaTokenizer")
    tokenizer: LlamaTokenizer = initialize_model_inference(model_name, token, return_model=False)  # type: ignore
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("initialize_model")
    model = initialize_model(
        model_name=model_name,
        token=token,
        base_class=trf.AutoModelForSequenceClassification,
    )
    logger.info("get_peft_model")
    model = peft.get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.print_trainable_parameters()
    # create datasets
    logger.info("generate_dataset")
    tr_data, val_data, train_path, valid_path = generate_dataset(
        folder_out=folder_out.parent,
        file_examples=file_examples,
        field_label=field_label,
        field_input=field_input,
        token=token,
        model_name=model_name,
        limit_tokens=limit_tokens,
        train_size=train_size,
        id=f"_{model_name}_{limit_tokens}",
    )
    logger.info(f"Using {train_path} {valid_path}")
    logger.info("dataloaders")

    def collate_fn(data: List[dict]):
        inputs = tokenizer(
            [d["text"] for d in data],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        # logger.info(
        #     f"batch {len(data)=} making of text of size {len(inputs[0])=} for text {data[0]['text']=}"
        # )
        outputs = [d[field_label] for d in data]
        return data, torch.tensor(inputs, dtype=torch.int32), torch.tensor(
            outputs, dtype=torch.float16
        )
    if lim_size == -1:
        lim_size = len(tr_data)
    train_dataloader = torch.utils.data.DataLoader(
        tr_data[:lim_size], shuffle=True, collate_fn=collate_fn, batch_size=tr_bs
    )
    if lim_size == -1:
        lim_size = len(val_data)
    eval_dataloader = torch.utils.data.DataLoader(
        val_data[:lim_size], shuffle=False, collate_fn=collate_fn, batch_size=val_bs
    )
    logger.info("training QLORA")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    # Instantiate scheduler
    lr_scheduler = trf.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * num_train_epochs),
        num_training_steps=(len(train_dataloader) * num_train_epochs),
    )
    
    if mapping_dict is None:
        mapping_dict = {
            -2: f"Too big >={limit_tokens}",
            -1: "Mixed answer",
            0: "NON SEVERE",
            1: "SEVERE",
        }
    device = "cuda"
    model.to(device)
    os.system("nvidia-smi")
    evaluator = Evaluator()
    assert num_train_epochs>0, "Train at least one epoch required"
    metrics_folder = folder_out / "metrics"
    metrics_folder.mkdir(exist_ok=True, parents=True)
    Ltr = []
    Lval = []
    for epoch in range(num_train_epochs):
        model.train()
        logger.info(f"{epoch=}")
        for step, (d, inputs, labels) in enumerate(tqdm.tqdm(train_dataloader)):
            # logger.info(f"{inputs.shape=}")
            # cpu_size_bytes = inputs.element_size() * inputs.numel()
            # logger.info(f"{cpu_size_bytes=}")
            inputs.to(device)
            # logger.info(f"after inputs.to(device)")
            inputs.to("cpu")
            # logger.info(f"after inputs.to('cpu')")
            inputs.to(device)
            # logger.info(f"after inputs.to(device)")
            outputs = model(inputs, return_dict=True)
            # logger.info(f"after model(inputs) {outputs=}")
            loss = outputs.loss['logits'].sum()
            Ltr.append({"loss":loss.tolist(),"step":step,"epoch":epoch,"tot_step":step+epoch*len(train_dataloader)})
            loss.backward()
            # logger.info(f"after backward")
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # logger.info(f"after step")

        model.eval()
        evaluator.reset()
        for step, (d, inputs, labels) in enumerate(tqdm.tqdm(eval_dataloader)):
            inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs, return_dict=True)
            Lval.append({"loss":outputs.loss['logits'].sum().tolist(),"step":step,"epoch":epoch,"tot_step":step+epoch*len(train_dataloader)})
            pred = outputs.logits.argmax(dim=-1).cpu().detach().tolist()
            logits = outputs.logits.detach().tolist()
            evaluator.add_samples(d=d,pred=pred,logits=logits)
        fields_data = evaluator.get_data()
        conf_matrix, f1, data = compute_metrics_from_list(
            fields_data, 
            tokenizer, 
            input_field="text", 
            pred_field="pred", 
            mapping_dict=mapping_dict, 
            n_tokens_infered_max=limit_tokens,
            n_tokens_show_max=limit_tokens, 
            n_tokens_show_min=0
        )
        accuracy = np.sum(np.diag(conf_matrix))/np.sum(conf_matrix)
        with open(metrics_folder / f"epoch_{epoch}.json", "w") as f:
            json.dump({
                "conf_matrix": conf_matrix.tolist(),
                "f1": f1,
                "data": data.to_dict(orient='records'),
                "Ltr":Ltr,
                "Lval":Lval
            },f,indent=2)
        print(f"epoch {epoch}: {accuracy=} {f1=}")
    
    logger.info("Saving trained QLORA model")
    if new_model_name != "":
        output_dir = folder_out / "trained_model"
        output_dir.mkdir()
        model.save_pretrained(str(output_dir))
        model.push_to_hub(new_model_name)
    return conf_matrix, f1, data #type: ignore


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
    logger.info("main_qlora_generation")
    if token != "":
        huggingface_hub.login(token=token)
    if not folder_out.exists():
        folder_out.mkdir(parents=True)

    logger.info("initialize_model_inference")
    tokenizer, model = initialize_model_inference(model_name, token, return_model=True)  # type: ignore
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("config elements")
    peft_config = peft.LoraConfig(  # type: ignore
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        inference_mode=False,
        task_type="CAUSAL_LM",
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
        eval_steps=eval_steps,
    )
    # create datasets
    logger.info("generate_dataset")
    train_data, val_data, train_path, valid_path = generate_dataset(
        folder_out=folder_out.parent,
        file_examples=file_examples,
        field_label=field_label,
        field_input=field_input,
        token=token,
        model_name=model_name,
        limit_tokens=limit_tokens,
        train_size=train_size,
        id=f"_{model_name}_{limit_tokens}",
    )
    logger.info(f"Using {train_path} {valid_path}")
    logger.info("load_dataset")
    for i in range(len(train_data)):
        train_data[i]['text'] += "\n"+str(train_data[i][field_label])
    for i in range(len(val_data)):
        val_data[i]['text'] += "\n"+str(val_data[i][field_label])
    train_dataset = datasets.Dataset.from_list(train_data)  # type: ignore
    valid_dataset = datasets.Dataset.from_list(val_data)  # type: ignore
    # Set supervised fine-tuning parameters
    logger.info("Set supervised fine-tuning parameters")
    trainer = trl.SFTTrainer(  # type: ignore
        model=model,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=valid_dataset,  # type: ignore
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        max_seq_length=limit_tokens,
    )
    logger.info("Starting training QLORA")
    trainer.train()
    logger.info("Saving trained QLORA model")
    if new_model_name != "":
        output_dir = folder_out / "trained_model"
        output_dir.mkdir()
        trainer.model.save_pretrained(output_dir)
        trainer.model.push_to_hub(new_model_name)


def get_max_tokens(
    evaluator: MaxTokensEvaluator,
    min_token_length: int = 0,
    max_token_length: int = 5000,
):
    """
    The main algorithm comes from get_max_mix.
    However dependency injection allows to provide the specific element to evaluate with MaxTokensEvaluator
    """

    min_token_length = max(1, min_token_length)  # because start token
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
            torch.cuda.empty_cache()  # type: ignore
        except Exception as e:
            print("Exception clear")
            print(e)
            print("End exception clear")
    return (max_work, min_not_work)

def get_pooling_operation(pooling_code: PoolingOperationCode) -> PoolingFn:
    if pooling_code == "mean":
        return lambda embedding: torch.mean(embedding,dim=0)
    elif pooling_code == "sum":
        return lambda embedding: torch.sum(embedding,dim=0)
    else:
        raise ValueError(f"{pooling_code=} is not possible")
    
@print_args
def get_llama2_embeddings(
    model_name: ModelName,
    path_data_preprocessed: Path,
    folder_out: Path,
    pooling_fn: 'PoolingFn',
    layers_ids: Optional[Tuple[int]] = None,
    start: int = 0,
    end: int = -1,
    limit_tokens: int = -1,
    id_pred: str = ""
):
    """From a json file with the description use llama2 to generate the embeddings for each data sample. The intent of this function is to be called with multiple nodes on a slurm server to have faster results
    
    # Arguments
    - model_name: ModelName, the name of the model to use to generate the embeddings
    - path_data_preprocessed: Path, the path to the json file containing the data in the format [{'description': "...", 'bug_id': ...}, ...]
    - folder_out, Path the folder where to put the data, name automatically determined with start seed
    - pooling_fn: PoolingFn, function to do the aggregation
    - layers_ids: Optional[Tuple[int]] = (0, ), the layers embeddings to use
    - start: int = 0, the starting element in the data to process
    - end: int = -1, the ending element to process in the data
    - limit_tokens: int = -1, the limit number of tokens to use (all by default)
    - id_pred: str = "", the id to put in the filename to help for the aggregation of files after
    
    """
    if layers_ids is None:
        layers_ids = (0,)
    tokenizer, model = initialize_model_inference(model_name, token, hidden_states=True)  # type: ignore
    with open(path_data_preprocessed) as f:
        data_preprocessed = json.load(f)
    data = data_preprocessed["data"]
    if end == -1:
        end = len(data)
    data = data[start:end]
    print(f"Running for {start=} {end=}")
    folder_predictions = folder_out
    folder_predictions.mkdir(exist_ok=True, parents=True)
    get_file_path = lambda idx_layer: folder_predictions / f"embeddings_chunk{id_pred}_layer_{idx_layer}_{start}.hdf5"
    for i, d in tqdm.tqdm(enumerate(data), total=len(data)):
        tokenized_full_text = tokenizer.encode(d["description"])
        tokenized_full_text = tokenized_full_text[:limit_tokens]
        embeddings = model(torch.tensor([tokenized_full_text], dtype=torch.int32))  # type: ignore
        for idx_layer in layers_ids:
            embedding = embeddings.hidden_states[idx_layer]
            pooled_embedding = pooling_fn(embedding).tolist()[0]
            with h5py.File(get_file_path(idx_layer), "a") as fp:
                fp.create_dataset(str(d['bug_id']),data=pooled_embedding,dtype="f")
        del embeddings
        gc.collect()
        torch.cuda.empty_cache()  # type: ignore

def merge_data_embeddings(
    folder_embeddings: Path,
    path_dst: Path,
    layer_id: int = -1,
    base_name: str = "embeddings_chunk_",
    auto_remove: bool = True
):
    """Allows to merge all hdf5 files of the embeddings into one. Automatically removes files after the data is transfered to save space.
    
    # Arguments 
    - folder_embeddings: Path, the path where are stored the embeddings
    - path_dst: Path, the path of the destination hdf5 file 
    - layer_id: int = -1, the layer id of the embedding to take
    - base_name: str = "embeddings_chunk_", the base name of the embedding hdf5 to merge
    - auto_remove: bool = True, if yes autoremove the files when the data have been transfered
    """
    sorted_path = list(folder_embeddings.rglob(f"{base_name}layer_{layer_id}_*.hdf5"))
    sorted_path = sorted(
        sorted_path, key=lambda x: int(x.name.split(".")[0].split("_")[-1])
    )
    with h5py.File(path_dst, "w") as fp:
        for p in sorted_path:
            print("Reading ", p)
            with h5py.File(p, "r") as fp:
                for (bug_id, embedding) in enumerate(fp.items()):
                    fp.create_dataset(bug_id, data=embedding)
            if auto_remove:
                p.unlink()

def get_classifier():
    # hidden_size = 64, random_state (train test split) = 0, batch_size = 32
    # criterion = nn.BCEWithLogitsLoss(), lr=0.001, num_epochs = 100
    binary_severities = []
    dict_data = []
    folder_path = Path(args.data_folder_path_to_save) / 'aggregation_files/'
    json_file_paths = [file for file in folder_path.glob("*.json")]

    for json_file_path in json_file_paths:
        with open(json_file_path, 'r') as f:
            for line in f:
                line_data = line.strip(",\n")
                data_string = line_data.replace("'", '"')

                data_string = data_string.replace("PosixPath(", "")
                data_string = data_string.replace(")", "")
                
                dict_data.append(eval(data_string))
                binary_severities.append(eval(data_string)['binary_severity'])
    train, test = skMsel.train_test_split(
        dict_data,
        test_size=0.2,
        random_state=0,
        stratify=binary_severities
    )
    def collate_fn(data: List[dict]):
        bug_ids = [d['bug_id'] for d in data]
        inputs = [d['aggregated_list'] for d in data]
        labels = [d['binary_severity'] for d in data]
        return torch.tensor(bug_ids, dtype=torch.float16), torch.tensor(inputs, dtype=torch.int16), torch.tensor(labels, dtype=torch.float16)

    # Define batch size and create a DataLoader
    batch_size = 32
    train_dataloader = dt.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    input_size = len(train[0]['aggregated_list']) #check
    hidden_size = 64
    output_size = 1

    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # lr = learning rate
    
    num_epochs = 100
    total_samples = len(train)
    for epoch in range(num_epochs):
        for i, (bug_ids, inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.reshape([-1,1]))
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_samples//batch_size+1}], Loss: {loss.item():.4f}')

    test_dataloader = dt.DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Set the model to evaluation mode
    model.eval()

    # bug_ids_list = []
    labels_list = []
    outputs_lists = []
    with torch.no_grad():
        for bug_ids, inputs, labels in test_dataloader:  # Iterate through your test dataset
            outputs = model(inputs)  # Forward pass
            predicted = (outputs > 0.5).float()
            # bug_ids_list.extend(bug_ids.tolist())
            labels_list.extend(labels.tolist())
            outputs_lists.extend(predicted.tolist())
            # with open(folder_path / 'predictions.json', 'a') as outfile:
            #     json.dump(prediction_data, outfile)
            #     outfile.write(',\n')


class DataoutDict(TypedDict):
    bug_id: str
    text: str
    answer: str
    severity_pred: Union[int, float]
    binary_severity: int

def aggr_finetune(
            folder_out: Path,
            folder_in: Path,
            pattern_name: str
        ):
    samples = []
    for p in folder_in.rglob(pattern=pattern_name):
        path_parameters = p / "parameters.json"
        path_metrics = p / "metrics"
        # get the parameters
        if not path_parameters.exists() or not path_metrics.exists():
            continue
        with open(path_parameters) as f:
            params = json.load(f)
        # extract the metrics
        for path_epoch in path_metrics.rglob("epoch_*"):
            epoch = int(path_epoch.stem.split("_")[-1])
            with open(path_epoch) as f:
                metrics = json.load(f)
            samples.append({
                **params,
                **metrics,
                "epoch":epoch
            })
    with open(folder_out / "metrics.json", "w") as f:
        json.dump(samples,f)
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
        "finetune_dataset",
        "finetune",
        "embeddings_gen",
        "nn_embedding",
        "nn_classifier",
        "aggr_finetune"
    ]
    parser.add_argument(
        "-path_data_json",
        type=path_check,
        help="Path to the json data file",
        default=f"/project/def-aloise/{os.environ['USER']}/data/data_preprocessed_tokens_v2.json",
    )
    parser.add_argument(
        "-path_data_folder",
        type=path_check,
        help="Root path to the main data folder",
        default=f"/project/def-aloise/{os.environ['USER']}/data/",
    )
    parser.add_argument(
        "-data_folder_path_to_save",
        type=path_check,
        help="Root path to the main data folder",
        default=f"/project/def-aloise/{os.environ['USER']}/data/",
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
        "-lr",
        type=float,
        help="Learning rate",
        default=1e-3,
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
        help="Base name of the json file with the layer id for get_data_embeddings (ex: embeddings_chunk__trunc_layer_-1_4460.h5 will give embeddings_chunk__trunc_)",
        default="embeddings_chunk_",
    )
    parser.add_argument(
        "-path_backup_fields",
        type=str,
        help="Allow to add the missing field of binary_severity based on the bug_id common field. Relative path to the data path",
        default="llm/data_preprocessed_tokens.json",
    )
    parser.add_argument(
        "-pattern_name",
        type=str,
        help="pattern of the root folders where the finetuning wrote metrics",
        default="qlora_finetune_class_*",
    )
    parser.add_argument(
        "-pooling_fn",
        choice=["mean","sum"],
        help="pooling function to use to do embeddings",
        default="mean",
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
        folder_out = args.path_data_folder / f"out_{args.pred_field}{args.id}"
        folder_out.mkdir(parents=True, exist_ok=True)
        
        tokenizer = initialize_model_inference(model_name, token, return_model=False)  # type: ignore
        fields_data = extract_fields_from_json(args.path_data_folder)
        conf_matrix, f1, data = compute_metrics_from_list(
            fields_data=fields_data, 
            tokenizer=tokenizer, 
            path_backup_fields=args.path_data_folder / args.path_backup_fields, 
            pred_field=args.pred_field,
            input_field=args.input_field,
            n_tokens_infered_max=args.n_tokens_infered_max,
            n_tokens_show_min=args.n_tokens_show_min,
            n_tokens_show_max=args.n_tokens_show_max,
        )
        compute_metrics_from_files(
            conf_matrix=conf_matrix,
            f1=f1,
            data_full=data,
            folder_out=folder_out,
            pred_field=args.pred_field,
            n_tokens_infered_max=args.n_tokens_infered_max,
            n_tokens_show_min=args.n_tokens_show_min,
            n_tokens_show_max=args.n_tokens_show_max,
        )
    elif args.algorithm == "finetune_dataset":
        generate_dataset(
            folder_out=args.path_data_folder,
            file_examples=args.path_data_json,
            file_split=args.path_data_folder / "split.json",
            token=args.token, 
            model_name=args.model_name,
            id=f"_{args.model_name}_{args.n_tokens_infered_max}",
            field_input=args.input_field,
            limit_tokens=args.n_tokens_infered_max,
        )
    elif args.algorithm == "finetune_generation":
        path_out = args.path_data_folder / f"qlora_finetune_{args.id}"
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
            num_train_epochs=args.num_train_epochs,
        )
    elif args.algorithm == "finetune_classification":
        path_out = args.path_data_folder / f"qlora_finetune_{args.id}"
        path_out.mkdir(parents=True, exist_ok=True)
        with open(path_out / "parameters.json", "w") as f:
            json.dump(vars(args),indent=4,fp=f,cls=CustomEncoder)
        conf_matrix, f1, data = main_qlora_classification(
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
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.lr
        )
        compute_metrics_from_files(
            conf_matrix=conf_matrix,
            f1=f1,
            data_full=data,
            folder_out=path_out,
            pred_field=args.pred_field,
            n_tokens_infered_max=args.n_tokens_infered_max,
            n_tokens_show_min=args.n_tokens_show_min,
            n_tokens_show_max=args.n_tokens_show_max,
        )
    elif args.algorithm == "max_tokens_pipeline":
        (max_work, min_not_work) = get_max_tokens(
            PipelineMaxTokens(model_name=args.model_name, token=args.token),
            min_token_length=1,
            max_token_length=10000,
        )
        logger.info(f"{max_work=},{min_not_work=}")
    elif args.algorithm == "max_tokens_embeddings":
        (max_work, min_not_work) = get_max_tokens(
            EmbeddingsMaxTokens(model_name=args.model_name, token=args.token),
            min_token_length=1,
            max_token_length=10000,
        )
        print(f"{max_work=},{min_not_work=}")
    elif args.algorithm == "max_tokens_finetuning":
        path_data_folder = Path(args.path_data_folder)
        if not path_data_folder.exists():
            raise ValueError(f"The path {path_data_folder} does not exist")
        max_n_tokens = 10000
        args_dict = convert_dict_to_str(vars(args))
        L = []
        for model_name in [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
        ]:
            for qlora_r in [2, 8, 16, 32, 64, 128, 256]:
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
                    max_token_length=max_n_tokens,
                )
                L.append(
                    {
                        "max_work": max_work,
                        "min_not_work": min_not_work,
                        "qlora_r": qlora_r,
                        "model_name": args.model_name,
                        "args": args_dict,
                    }
                )
                with open(path_data_folder / "finetune_tokens_lim.json", "w") as f:
                    json.dump(L, f)
    elif args.algorithm == "embeddings_gen":
        folder_out = args.path_data_folder / "embeddings"
        folder_out.mkdir(parents=True,exist_ok=True)
        layers_ids = eval(args.layers_ids)
        pooling_fn = get_pooling_operation(args.pooling_fn)
        get_llama2_embeddings(
            model_name=args.model_name,
            folder_out=folder_out,
            path_data_preprocessed=args.path_data_json,
            layers_ids=layers_ids,
            start=seed_start,
            end=seed_end,
            id_pred=args.id,
            limit_tokens=args.n_tokens_infered_max,
            pooling_fn=pooling_fn
        )
    elif args.algorithm == "embeddings_agg":
        folder_embeddings = args.path_data_folder / "embeddings"
        layers_ids = eval(args.layers_ids)
        merge_data_embeddings(
            folder_embeddings=folder_embeddings,
            path_dst=args.path_data_folder / "embeddings.hdf5",
            layer_id=layers_ids[0],
            base_name=args.base_name
        )
    elif args.algorithm == "aggr_finetune":
        folder_out: Path = args.path_data_folder / "train_class"
        folder_out.mkdir(exist_ok=True, parents=True)
        aggr_finetune(
            folder_out=folder_out,
            folder_in=args.path_data_folder,
            pattern_name=args.pattern_name
        )
    elif args.algorithm == "nn_embedding":
        folder_embeddings = Path(args.path_data_folder) / "embeddings"
        layer_id: Tuple[int] = eval(args.layers_ids)
        if len(layer_id) > 1:
            raise ValueError(f"Expecting just one layer id not {len(layer_id)}")
        print(args.algorithm)

        with open(Path(args.data_folder_path_to_save) / 'output_file.json', 'w') as outfile:
            outfile.write("")

        for d in get_data_embeddings(
            folder_embeddings=folder_embeddings,
            layer_id=layer_id[0],
            base_name=args.base_name,
        ):
            bug_id = d["bug_id"]
            binary_severity = d["binary_severity"]
            hidden_state = np.array(d["hidden_state"])

            base_name = args.base_name
            came_from = f"{base_name}layer_{layer_id[0]}_.json"

            sum_aggregated_array = np.sum(hidden_state, axis=0)
            mean_aggregated_array = sum_aggregated_array / len(hidden_state)

            aggregated_list = mean_aggregated_array.tolist()

            data = {
                "bug_id": bug_id,
                "binary_severity": binary_severity,
                "from": came_from,
                "aggregated_list": aggregated_list,
            }
            with open(Path(args.data_folder_path_to_save) / 'output_aggregation.json', 'a') as outfile:
                json.dump(data, outfile)
                outfile.write(",\n")
    elif args.algorithm == "nn_classifer":
        get_classifier()