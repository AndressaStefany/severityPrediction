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
from typing import Any, Dict, List, Optional, Tuple, Union
import psutil
import multiprocessing as mp
import functools
import random
import fire
import torch
from torch import nn
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

# tye hints
LlamaTokenizer = Union["trf.LlamaTokenizer", "trf.LlamaTokenizerFast"]
LlamaModel = "trf.LlamaForCausalLM"
PoolingOperationCode = Literal["mean", "sum"]
PoolingFn = Callable[["torch.Tensor"], "torch.Tensor"]
ModelName = Literal["meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf"]
DatasetName = Literal["eclipse_72k", "mozilla_200k"]
BugId: int
default_token: str = "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"
default_model: ModelName = "meta-llama/Llama-2-13b-chat-hf"
default_n_tokens_infered_max: int = 7364
default_input_field: str = "description"
default_folder_data: Path = Path(f"/project/def-aloise/{os.environ['USER']}/data")

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
    import bitsandbytes as bnb
    import evaluate
    import optuna


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
    "import bitsandbytes as bnb",
    "import evaluate",
    "import optuna",
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


def existing_path(p: Union[str, Path], is_folder: bool) -> Path:
    p = Path(p)
    if not p.exists():
        raise Exception(f"{p.resolve()} does not exists")
    if p.is_dir() and not is_folder:
        raise Exception(f"{p.resolve()} is a folder not a file")
    if not p.is_dir() and is_folder:
        raise Exception(f"{p.resolve()} is a file not a folder")
    return p


def assert_valid_token(token: str):
    assert isinstance(token, str) and len(token) > 3 and token[:3] == "hf_"


def get_literal_value(model_name: str, literal: Any = ModelName) -> Any:
    assert isinstance(model_name, str) and model_name in get_args(literal)
    return model_name  # type: ignore


def get_dataset_choice(dataset_choice: str) -> DatasetName:
    assert isinstance(dataset_choice, str) and dataset_choice in get_args(DatasetName)
    return dataset_choice  # type: ignore


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.leaky_relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out


class PreprocessedData(TypedDict, total=True):
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


class EmbeddingDictElem(TypedDict, total=True):
    """Contains especially
    - bug_id: int, the id of the bug
    - binary_severity: int, the severity level of the bug 0=NON SEVERE, 1=SEVERE
    - description: str, the field that has been used to generate the embeddings
    - stemmed_description: str, the field that has been used to generate the embeddings
    - hidden_state: np.ndarray of shape (size_vocab,), the embedding
    """

    bug_id: int
    binary_severity: int
    description: str
    stemmed_description: str
    hidden_state: "np.ndarray"


class EmbeddingDict(TypedDict, total=True):
    tr: List[EmbeddingDictElem]
    val: List[EmbeddingDictElem]
    test: List[EmbeddingDictElem]


class DataDict(TypedDict, total=True):
    tr: List[PreprocessedData]
    val: List[PreprocessedData]
    test: List[PreprocessedData]


class SplitDict(TypedDict, total=True):
    """Contains the bug_ids of the data for the train validation and test set"""

    tr: List[int]
    val: List[int]
    test: List[int]


class CustomFormatter(logging.Formatter):
    def __init__(
        self, fmt=None, datefmt=None, style="%", validate: bool = True
    ) -> None:
        super().__init__(fmt, datefmt, style, validate)
        try:
            self.total_ram_gpu = float(
                subprocess.check_output(
                    "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0",
                    shell=True,
                )
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
        template_path: Path,
        n_samples: int = 100,
        **kwargs_train,
    ) -> None:
        super().__init__()
        tokenizer = get_tokenizer(token, model_name)
        self.kwargs = {"model_name": model_name, "token": token, **kwargs_train}
        token_ids = generate_text_with_n_tokens(tokenizer, max_n_tokens)
        token_names = tokenizer.convert_ids_to_tokens(token_ids)
        with open(template_path) as f:
            template = json.load(f)
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
        trust_remote_code=True,
    )  # type: ignore
    return tokenizer


def initialize_model(
    model_name: str,
    token: str,
    hidden_states: bool = False,
    base_class: Any = trf.AutoModelForCausalLM,
    num_labels: int = 1,
    quant: bool = True,
    *args, **kwargs
) -> "trf.LlamaForCausalLM":
    huggingface_hub.login(token=token)
    double_quant_config = None
    if quant:
        double_quant_config = trf.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
    model = base_class.from_pretrained(
        model_name,
        quantization_config=double_quant_config,
        return_dict=hidden_states,
        output_hidden_states=hidden_states,
        cache_dir="/project/def-aloise/rmoine/cache_dir",
        token=token,
        num_labels=num_labels,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def initialize_model_inference(
    model_name: str,
    token: str,
    return_model: bool = True,
    hidden_states: bool = False,
    num_labels: int = 1,
) -> Union[Tuple[LlamaTokenizer, "trf.LlamaForCausalLM"], LlamaTokenizer]:
    huggingface_hub.login(token=token)
    tokenizer = get_tokenizer(token=token, model_name=model_name)
    if return_model:
        model = initialize_model(
            model_name,
            token,
            hidden_states,
            trf.AutoModelForCausalLM,
            num_labels=num_labels,
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
    path_data_json: str,  # type: ignore
    model_name: str = default_model,
    token: str = default_token,
    n_tokens_infered_max: int = 7364,
    id_pred: str = "",
    n_data: Optional[int] = None,
    seed_start: Optional[int] = None,
    seed_end: Optional[int] = None,
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
):
    path_data_json: Path = existing_path(path_data_json, is_folder=False)
    assert model_name in get_args(
        ModelName
    ), f"Expecting model name to be in {get_args(ModelName)} not {model_name}"
    assert len(token) > 3 and token[:3] == "hf_"
    seed_start, seed_end = generate_seeds(
        n_data, seed_start, seed_end, n_chunks, interval_idx
    )

    tokenizer, model = initialize_model_inference(model_name, token)  # type: ignore
    pipeline = trf.pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
    )
    with open(path_data_json) as f:
        data_preprocessed = json.load(f)
    data = data_preprocessed["data"]
    template = data_preprocessed["template"]
    data = data[seed_start:seed_end]
    responses = []
    folder_predictions = path_data_json.parent / "predictions"
    folder_predictions.mkdir(exist_ok=True, parents=True)
    with open(folder_predictions / f"metadata.meta", "w") as f:
        json.dump({"data_path": str(path_data_json.resolve())}, f, indent=2)
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
            limit_tokens=n_tokens_infered_max,
        )
        n_tokens = len(tokenized_full_text)
        assert n_tokens < n_tokens_infered_max
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
                folder_predictions
                / f"predictions_v100l_chunk_{seed_start}_{id_pred}.json",
                "w",
            ) as f:
                json.dump(responses, f)

    with open(
        folder_predictions / f"predictions_v100l_chunk_{seed_start}_{id_pred}.json", "w"
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
    mapping_dict: Optional[dict] = None,
):
    data = pd.DataFrame(fields_data)
    assert len(data) > 0
    # Remove duplicates by bug_id
    data.drop_duplicates(subset="bug_id", inplace=True)
    assert len(data) > 1
    if "binary_severity" not in data.columns:
        if path_backup_fields is not None:
            df_bs = pd.read_json(path_backup_fields)
            # check that we have the same bug_ids in the two dataframes
            assert len(data[data["bug_id"].isin(df_bs)]) == len(
                data
            ), "Expecting to have all bug_ids of predictions in the backup file"
            data = data.merge(
                df_bs[["bug_id", "binary_severity"]], on="bug_id", how="left"
            )
        else:
            raise Exception(f"Missing field binary_severity having {data.columns}")
    # Count number of tokens
    if tokenizer is not None:
        data["n_tokens"] = data[input_field].apply(lambda x: len(tokenizer(x)["input_ids"]))  # type: ignore
    data_full = data.copy()
    # Filter by limit of tokens
    if "n_tokens" in data.columns:
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
    backend: Optional[str] = "agg",
    id: str = "",
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
        print(data_full)
        data_full.to_json(folder_out / f"data{id}.json", orient="records", indent=4)
    with open(folder_out / "metrics{id}.json", "w") as f:
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
                set(data_full[true_field].unique().tolist()).union(
                    set(data_full[pred_field].unique().tolist())
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
            field_pred=pred_field,
            field_true=true_field,
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
    df: "pd.DataFrame",
    path_out: Path,
    mapping_dict: Dict,
    n_samples: int = 5,
    field_pred: str = "pred",
    field_true: str = "true",
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
    poss = df[field_pred].unique().tolist()
    for i, j in product(poss, poss):
        df_sel = (
            df.query(f"{field_pred} == {i} & {field_true} == {j}")
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
        for d_e, pred_e, logits_e in zip(d, pred, logits):
            d_saved = {**d_e, "pred": pred_e, "logits": logits_e}
            self.buffer.append(d_saved)

    def get_data(self):
        return self.buffer


@print_args
def generate_dataset(
    folder_out: Union[str, Path],  # type: ignore
    folder_data: Union[str, Path],  # type: ignore
    dataset_choice: DatasetName,
    field_input: str,
    token: str = default_token,
    model_name: ModelName = default_model,
    n_tokens_infered_max: int = -1,
    id: str = "",
):
    """Generates the dataset for the finetuning"""
    folder_out: Path = existing_path(folder_out, is_folder=True)
    folder_data: Path = existing_path(folder_data, is_folder=True)
    file_examples = existing_path(
        folder_data / f"{dataset_choice}.json", is_folder=False
    )
    file_split = existing_path(
        folder_data / f"split_{dataset_choice}.json", is_folder=False
    )
    assert_valid_token(token)
    model_name = get_literal_value(model_name)

    logger.info("Create datasets")
    id = id.replace("/", "_")
    train_path = folder_out / f"finetune_train{id}.json"
    valid_path = folder_out / f"finetune_valid{id}.json"
    full_path = folder_out / f"finetune_full{id}.json"
    if not train_path.exists() or not valid_path.exists():
        tokenizer = get_tokenizer(token, model_name)
        logger.info("-> Creating cache")
        with open(file_examples) as f:
            data_preprocessed = json.load(f)
        data: List[PreprocessedData] = data_preprocessed
        with open(file_examples.parent / "template.json") as fp:
            template = json.load(fp)
        with open(file_split) as f:
            splits: SplitDict = {k: set(v) for k, v in json.load(f).items()}  # type: ignore
        L = {"tr": [], "val": [], "test": []}
        for d in tqdm.tqdm(data):
            tokens_ids: List[int] = tokenizer(d[field_input])["input_ids"]  # type: ignore
            tokenized: List[str] = tokenizer.convert_ids_to_tokens(tokens_ids)  # type: ignore
            text, tokenized_full_text = build_prompt(
                template["llama_tokenized_template"],
                tokenized,
                template["template_index_insert"],
                tokenizer,
                limit_tokens=n_tokens_infered_max,
            )
            d = {**d, "input": text, "n_tokens": len(tokenized_full_text)}
            for k in ["tr", "val", "test"]:
                if d["bug_id"] in splits[k]:
                    L[k].append(d)
                    break
        logger.info("-> End creation cache in memory")
        with open(full_path, "w") as f:
            json.dump(L, f)
        with open(train_path, "w") as f:
            json.dump(L["tr"], f)
        with open(valid_path, "w") as f:
            json.dump(L["val"], f)
        return L["tr"], L["val"], train_path, valid_path
    else:
        logger.info("-> Reading cache")
        with open(train_path, "r") as f:
            train_data = json.load(f)
        with open(valid_path, "r") as f:
            valid_data = json.load(f)
        return train_data, valid_data, train_path, valid_path


class LossAggregator(trf.trainer_callback.TrainerCallback):
    def __init__(self, event: Literal["train", "val"]) -> None:
        self.aggregator = {"loss": 0.0, "num_samples": 0}
        self.history = []
        self.event = event
        super().__init__()

    def set_loss_fn(self, loss_fn: str):
        self.loss_fn = loss_fn

    def add_new_data(
        self,
        bug_id: List[int],
        predictions: List[int],
        trues: List[int],
        loss: float,
        num_samples: int,
        n_tokens: List[int],
        event: Literal["val", "train"],
    ):
        if event in self.event:
            self.aggregator["loss"] += loss
            self.aggregator["num_samples"] += num_samples

    def gather_epoch(
        self,
        args: "trf.TrainingArguments",
        state: "trf.TrainerState",
        control: "trf.TrainerControl",
        **kwargs,
    ):
        if state.epoch < 1:
            return
        loss_dict: dict = {**self.aggregator}
        self.aggregator = {"loss": 0.0, "num_samples": 0}
        loss_dict["epoch"] = state.epoch
        self.history.append(loss_dict)

    def on_epoch_end(
        self,
        args: "trf.TrainingArguments",
        state: "trf.TrainerState",
        control: "trf.TrainerControl",
        **kwargs,
    ):
        if "train" == self.event:
            logger.info(f"gather_epoch {state.epoch} {self.event}")
            self.gather_epoch(args, state, control, **kwargs)
        return super().on_epoch_end(args, state, control, **kwargs)

    def on_evaluate(
        self,
        args: "trf.TrainingArguments",
        state: "trf.TrainerState",
        control: "trf.TrainerControl",
        **kwargs,
    ):
        if "val" == self.event:
            logger.info(f"gather_epoch {state.epoch} {self.event}")
            self.gather_epoch(args, state, control, **kwargs)
        return super().on_evaluate(args, state, control, **kwargs)


class PredictionAggregator(trf.trainer_callback.TrainerCallback):
    def __init__(
        self,
        event: Literal["train", "val"],
        n_tokens_infered_max: int,
        folder_out: Path,
    ) -> None:
        self.history = []
        self.epoch_buffer = []
        self.event = event
        self.n_tokens_infered_max = n_tokens_infered_max
        self.folder_out = folder_out
        super().__init__()

    def add_new_data(
        self,
        bug_ids: List[int],
        predictions: List[int],
        trues: List[int],
        loss: float,
        num_samples: int,
        n_tokens: List[int],
        event: Literal["val", "train"],
    ):
        for bug_id, prediction, true, n in zip(bug_ids, predictions, trues, n_tokens):
            self.epoch_buffer.append(
                {
                    "bug_id": bug_id,
                    "prediction": int(np.round(prediction)),
                    "probability": prediction,
                    "binary_severity": true,
                    "n_tokens": n,
                }
            )

    def gather_epoch(
        self,
        args: "trf.TrainingArguments",
        state: "trf.TrainerState",
        control: "trf.TrainerControl",
        **kwargs,
    ):
        if state.epoch < 1 or len(self.epoch_buffer) == 0:
            return
        for i in range(len(self.epoch_buffer)):
            self.epoch_buffer[i]["epoch"] = state.epoch
        conf_matrix, f1, data_full = compute_metrics_from_list(
            fields_data=self.epoch_buffer,
            pred_field="prediction",
            n_tokens_infered_max=self.n_tokens_infered_max,
        )
        compute_metrics_from_files(
            conf_matrix=conf_matrix,
            f1=f1,
            folder_out=self.folder_out,
            data_full=data_full,
            true_field="binary_severity",
            pred_field="prediction",
            n_tokens_infered_max=self.n_tokens_infered_max,
            n_tokens_show_max=self.n_tokens_infered_max,
            id=f"_epoch_{str(state.epoch).replace('.','-')}_{self.event}",
        )
        logger.info(f"{conf_matrix=}")
        self.epoch_buffer = []

    def on_epoch_end(
        self,
        args: "trf.TrainingArguments",
        state: "trf.TrainerState",
        control: "trf.TrainerControl",
        **kwargs,
    ):
        if "train" == self.event:
            logger.info(f"gather_epoch {state.epoch} {self.event}")
            self.gather_epoch(args, state, control, **kwargs)
        return super().on_epoch_end(args, state, control, **kwargs)

    def on_evaluate(
        self,
        args: "trf.TrainingArguments",
        state: "trf.TrainerState",
        control: "trf.TrainerControl",
        **kwargs,
    ):
        if "val" == self.event:
            logger.info(f"gather_epoch {state.epoch} {self.event}")
            self.gather_epoch(args, state, control, **kwargs)
        return super().on_evaluate(args, state, control, **kwargs)


class CustomTrainer(trl.SFTTrainer):
    def __init__(self, tokenizer, callbacks: List, *args, **kwargs):
        self.tokenizer = tokenizer
        self.loss_fn = torch.nn.functional.binary_cross_entropy
        self.callbacks = callbacks
        super().__init__(callbacks=callbacks, *args, **kwargs)

    def prediction_step(
        self, model, inputs, prediction_loss_only: bool, ignore_keys=None
    ) -> Tuple:
        logger.info(f"prediction_step with batch size of {len(inputs['bug_id'])}")
        try:
            gc.collect()
            torch.cuda.empty_cache()  # type: ignore
        except Exception as e:
            print("Exception clear")
            print(e)
            print("End exception clear")
        input = inputs["input"]
        pad_token = self.tokenizer(self.tokenizer.pad_token)["input_ids"][1]
        n_tokens = [len([e for e in elem if e != pad_token]) for elem in input.tolist()]
        label = inputs["label"]
        bug_id = inputs["bug_id"]
        prediction = model(input)[0]
        prediction = torch.nn.functional.sigmoid(prediction)
        loss = self.loss_fn(
            input=prediction,
            target=label,
        )
        predictions = prediction.reshape((-1,)).tolist()
        trues = label.reshape((-1,)).tolist()
        for c in self.callbacks:
            c.add_new_data(
                bug_id,
                predictions=predictions,
                trues=trues,
                loss=loss.sum().item(),
                n_tokens=n_tokens,
                event="pred",
                num_samples=len(bug_id),
            )
        return loss, predictions, trues  # to save GPU RAM

    def compute_loss(self, model, inputs, *args, **kwargs):
        logger.info(f"compute_loss with batch size of {len(inputs['bug_id'])}")
        try:
            gc.collect()
            torch.cuda.empty_cache()  # type: ignore
        except Exception as e:
            print("Exception clear")
            print(e)
            print("End exception clear")
        input = inputs["input"]
        pad_token = self.tokenizer(self.tokenizer.pad_token)["input_ids"][1]
        n_tokens = [len([e for e in elem if e != pad_token]) for elem in input.tolist()]
        label = inputs["label"]
        bug_id = inputs["bug_id"]
        prediction = model(input)[0]
        prediction = torch.nn.functional.sigmoid(prediction)
        loss = self.loss_fn(
            input=prediction,
            target=label,
        )
        predictions = prediction.reshape((-1,)).tolist()
        trues = label.reshape((-1,)).tolist()
        for c in self.callbacks:
            c.add_new_data(
                bug_id,
                predictions=predictions,
                trues=trues,
                loss=loss.sum().item(),
                n_tokens=n_tokens,
                event="train",
                num_samples=len(bug_id),
            )
        return loss


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        l_dict: List[dict],
        input_field: str = "input",
        label_field: str = "binary_severity",
    ):
        self.l_dict = l_dict
        self.input_field = input_field
        self.label_field = label_field
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.l_dict)

    def __getitem__(self, i: int) -> Dict[str, "torch.Tensor"]:
        elem = {**self.l_dict[i]}
        input = elem[self.input_field]
        label = torch.tensor(elem[self.label_field])
        bug_id = elem["bug_id"]
        return {"input": input, "label": label, "bug_id": bug_id}


class DataCollator(trf.data.DataCollatorForTokenClassification):
    def __init__(self, tokenizer, padding: bool, max_length: int):
        self.token_pad = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        tokenizer.pad_token = "[PAD]"
        super().__init__(tokenizer=tokenizer, padding=padding, max_length=max_length)

    def torch_call(self, features) -> Dict:
        logger.info(f"bs={len(features)}")
        inputs = [e["input"] for e in features]
        inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")["input_ids"]
        labels = torch.cat(
            [e["label"].reshape((1, 1)).to(torch.float) for e in features], dim=0
        )
        bug_id = [e["bug_id"] for e in features]
        return {"input": inputs, "label": labels, "bug_id": bug_id}


@print_args
def main_qlora_classification(
    dataset_choice: DatasetName,
    folder_out: Path = default_folder_data,
    folder_data: Path = default_folder_data,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_r: int = 64,
    model_name: str = default_model,
    token: str = default_token,
    field_input: str = "description",
    num_train_epochs: int = 1,
    tr_bs: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-4,
    limit_tokens: int = 7364,
    mapping_dict: Optional[dict] = None,
    lim_size: int = 500,
    id: str = "",
    use_cpu: bool = False,
) -> Tuple["np.ndarray", List[float], "pd.DataFrame"]:
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
    arguments = locals()
    logger.info("main_qlora")
    folder_out = existing_path(folder_out, is_folder=True) / f"qlora_finetune{id}"
    folder_out.mkdir(parents=True, exist_ok=True)
    folder_data = existing_path(folder_data, is_folder=True)
    assert_valid_token(token)
    model_name = get_literal_value(model_name)
    if token != "":
        huggingface_hub.login(token=token)
    with open(folder_out / "parameters.json", "w") as f:
        json.dump(arguments, indent=4, fp=f, cls=CustomEncoder)

    logger.info("LlamaTokenizer")
    tokenizer: LlamaTokenizer = get_tokenizer(model_name=model_name, token=token)
    logger.info("initialize_model")
    model = initialize_model(
        model_name=model_name,
        token=token,
        base_class=trf.AutoModelForSequenceClassification,
        quant=not use_cpu,
    )
    logger.info("peft.LoraConfig")
    peft_config = peft.LoraConfig(  # type: ignore
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        inference_mode=False,
        task_type="SEQ_CLS",
    )
    tokenizer.pad_token = "[PAD]"
    model.config.pad_token_id = 0
    # https://github.com/huggingface/transformers/blob/ce2e7ef3d96afaf592faf3337b7dd997c7ad4928/src/transformers/models/llama/modeling_llama.py#L906
    logger.info("get_peft_model")
    # model = peft.get_peft_model(model, peft_config)# type: ignore
    # model.print_trainable_parameters()
    print(model)
    # create datasets
    logger.info("generate_dataset")
    tr_data, val_data, train_path, valid_path = generate_dataset(
        folder_out=folder_data,
        folder_data=folder_data,
        field_input=field_input,
        dataset_choice=dataset_choice,
        token=token,
        model_name=model_name,  # type: ignore
        id=f"_{model_name}{id}_{limit_tokens}",
        n_tokens_infered_max=limit_tokens,
    )
    logger.info(f"Using {train_path} {valid_path}")
    logger.info("dataloaders")
    real_lim_size_tr = lim_size
    if lim_size == -1:
        real_lim_size_tr = len(tr_data)
    tr_data = Dataset(tokenizer, tr_data[:real_lim_size_tr])
    real_lim_size_val = lim_size
    if lim_size == -1:
        real_lim_size_val = len(val_data)
    val_data = Dataset(tokenizer, val_data[:real_lim_size_val])
    logger.info("training QLORA")
    # Set training parameters
    training_arguments = trf.TrainingArguments(
        output_dir=str(folder_out.resolve()),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=tr_bs,
        per_device_eval_batch_size=tr_bs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=1,
        learning_rate=learning_rate,
        fp16=not use_cpu,
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        # eval_accumulation_steps=1,
        logging_first_step=True,
        use_cpu=use_cpu,
    )

    if mapping_dict is None:
        mapping_dict = {
            -2: f"Too big >={limit_tokens}",
            -1: "Mixed answer",
            0: "NON SEVERE",
            1: "SEVERE",
        }
    logger.info("Set supervised fine-tuning parameters")
    loss_aggregator_tr = LossAggregator(event="train")
    predictions_aggregator_tr = PredictionAggregator(
        event="train", n_tokens_infered_max=limit_tokens, folder_out=folder_out
    )
    loss_aggregator_val = LossAggregator(event="val")
    predictions_aggregator_val = PredictionAggregator(
        event="val", n_tokens_infered_max=limit_tokens, folder_out=folder_out
    )
    trainer = CustomTrainer(  # type: ignore
        model=model,
        train_dataset=tr_data,
        eval_dataset=val_data,
        peft_config=peft_config,  # type: ignore
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        max_seq_length=limit_tokens + 5,
        formatting_func=lambda x: x,
        data_collator=DataCollator(tokenizer, False, limit_tokens),
        callbacks=[
            loss_aggregator_tr,
            predictions_aggregator_tr,
            loss_aggregator_val,
            predictions_aggregator_val,
        ],
    )
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)  # type :ignore
    logger.info(f"{trainer.args._n_gpu=}")

    trainer.train()
    with open(folder_out / "log_history.json", "w") as fp:
        json.dump(trainer.state.log_history, fp)


@print_args
def main_qlora_generation(
    file_examples: Path,
    file_split: Path,
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
        file_split=file_split,
        field_input=field_input,
        token=token,
        model_name=model_name,
        limit_tokens=limit_tokens,
        id=f"_{model_name}_{limit_tokens}",
    )
    logger.info(f"Using {train_path} {valid_path}")
    logger.info("load_dataset")
    for i in range(len(train_data)):
        train_data[i]["input"] += "\n" + str(train_data[i][field_label])
    for i in range(len(val_data)):
        val_data[i]["input"] += "\n" + str(val_data[i][field_label])
    train_dataset = datasets.Dataset.from_list(train_data)  # type: ignore
    valid_dataset = datasets.Dataset.from_list(val_data)  # type: ignore
    # Set supervised fine-tuning parameters
    logger.info("Set supervised fine-tuning parameters")
    trainer = CustomTrainer(  # type: ignore
        model=model,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=valid_dataset,  # type: ignore
        peft_config=peft_config,  # type: ignore
        dataset_text_field="input",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        max_seq_length=limit_tokens + 10,
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


def get_pooling_operation(pooling_code: "PoolingOperationCode") -> "PoolingFn":
    if pooling_code == "mean":
        return lambda embedding: torch.mean(embedding, dim=0)
    elif pooling_code == "sum":
        return lambda embedding: torch.sum(embedding, dim=0)
    else:
        raise ValueError(f"{pooling_code=} is not possible")


@print_args
def get_llama2_embeddings(
    folder_out: str,  # type: ignore
    folder_data: str,  # type: ignore
    dataset_choice: DatasetName,
    pooling_op: PoolingOperationCode,
    layer_id: int = -1,
    seed_start: Optional[int] = None,
    seed_end: Optional[int] = None,
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
    limit_tokens: int = -1,
    id_pred: str = "",
    model_name: ModelName = default_model,  # type: ignore
    token: str = default_token,
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
    id_pred = id_pred.replace("/", "--")
    folder_out: Path = (
        existing_path(folder_out, is_folder=True) / f"embeddings{id_pred}"
    )
    folder_out.mkdir(parents=True, exist_ok=True)
    folder_data: Path = existing_path(folder_data, is_folder=True)
    path_data_preprocessed: Path = existing_path(
        folder_data / f"{dataset_choice}.json", is_folder=False
    )
    assert_valid_token(token)
    model_name: ModelName = get_literal_value(model_name)
    pooling_fn: PoolingFn = get_pooling_operation(
        get_literal_value(pooling_op, PoolingOperationCode)
    )
    tokenizer, model = initialize_model_inference(model_name, token, hidden_states=True)  # type: ignore
    with open(path_data_preprocessed) as f:
        data_preprocessed = json.load(f)
    start, end = generate_seeds(
        n_data=len(data_preprocessed),
        seed_start=seed_start,
        seed_end=seed_end,
        n_chunks=n_chunks,
        interval_idx=interval_idx,
    )
    data = data_preprocessed
    if end == -1:
        end = len(data)
    data = data[start:end]
    path_missing = folder_out / f"missing{id_pred}_{start}.json"
    get_file_path = (
        lambda layer_id: folder_out
        / f"embeddings_chunk{id_pred}_layer_{layer_id}_{start}.hdf5"
    )
    if not path_missing.exists() and get_file_path(layer_id).exists():
        logger.info("Nothing to do, file is already here and no missing")
        return
    if path_missing.exists():
        logger.info("Restart from missing")
        with open(path_missing) as fp:
            data = json.load(fp)

    print(f"Running for {start=} {end=}")
    folder_predictions = folder_out
    folder_predictions.mkdir(exist_ok=True, parents=True)
    Lmissing = []
    for i, d in tqdm.tqdm(enumerate(data), total=len(data)):
        tokenized_full_text = tokenizer.encode(d["description"])
        limit_tokens_sample = limit_tokens
        if limit_tokens == -1:
            limit_tokens_sample = len(tokenized_full_text)
        tokenized_full_text = tokenized_full_text[:limit_tokens_sample]
        logger.info(f"{len(tokenized_full_text)=}")
        try:
            embeddings = model(torch.tensor([tokenized_full_text], dtype=torch.int32))  # type: ignore
            embedding = embeddings.hidden_states[layer_id]
            pooled_embedding = np.array(
                pooling_fn(embedding).tolist()[0], dtype=np.float32
            )
            with h5py.File(get_file_path(layer_id), "a") as fp:
                fp.create_dataset(str(d["bug_id"]), data=pooled_embedding, dtype="f")
            del embeddings
        except torch.cuda.OutOfMemoryError:
            logger.info(f"Error for {len(tokenized_full_text)} tokens")
            Lmissing.append(d)
        gc.collect()
        torch.cuda.empty_cache()  # type: ignore
    with open(path_missing, "w") as fp:
        json.dump(Lmissing, fp)


@print_args
def merge_data_embeddings(
    folder_embeddings: Path,
    path_dst: Path,
    layer_id: int = -1,
    base_name_hdf5: str = "embeddings_chunk_",
    base_name_missing: str = "embeddings_chunk_",
    auto_remove: bool = False,
):
    """Allows to merge all hdf5 files of the embeddings into one. Automatically removes files after the data is transfered to save space.

    # Arguments
    - folder_embeddings: Path, the path where are stored the embeddings
    - path_dst: Path, the path of the destination hdf5 file
    - layer_id: int = -1, the layer id of the embedding to take
    - base_name: str = "embeddings_chunk_", the base name of the embedding hdf5 to merge
    - auto_remove: bool = True, if yes autoremove the files when the data have been transfered
    """
    if isinstance(layer_id, str):
        layer_id = eval(layer_id)[0]
    if isinstance(layer_id, tuple):
        layer_id = layer_id[0]
    path_dst = Path(path_dst)
    folder_embeddings = existing_path(folder_embeddings, is_folder=True)
    sorted_path = list(
        folder_embeddings.rglob(f"{base_name_hdf5}layer_{layer_id}_*.hdf5")
    )
    assert len(sorted_path) > 0, f"Nothing found in {folder_embeddings.resolve()}"
    sorted_path = sorted(
        sorted_path, key=lambda x: int(x.name.split(".")[0].split("_")[-1])
    )
    logger.info(f"{sorted_path=}")
    with h5py.File(path_dst, "w") as fp:
        for p in sorted_path:
            print("Reading ", p)
            with h5py.File(p, "r") as fp2:
                for i, (bug_id, embedding) in tqdm.tqdm(
                    enumerate(fp2.items()), total=len(fp2)
                ):
                    data = np.copy(embedding)
                    fp.create_dataset(str(bug_id), data=data, dtype="f")
            if auto_remove:
                p.unlink()
    sorted_path = list(folder_embeddings.rglob(f"{base_name_missing}*.json"))
    assert (
        len(sorted_path) > 0
    ), f"Expecting pathes for {folder_embeddings.resolve()} and {base_name_missing}"
    L = []
    for p in sorted_path:
        print("Reading json ", p)
        with open(p) as fp:
            L.extend(json.load(fp))
        if auto_remove:
            p.unlink()
    path_json = path_dst.parent / f"{base_name_missing}_missing.json"
    logger.info(f"{path_json=}")
    with open(path_json, "w") as fp:
        json.dump(L, fp)


def get_nn_classifier(trial: "optuna.Trial", input_size, output_size: int = 1):
    activation_functions = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "linear": nn.Linear,
    }
    n_layers = trial.suggest_int("n_layers", 3, 10)
    layers = []
    in_features = input_size

    out_features = trial.suggest_int(f"n_units_l0", 4, 128)
    layers.append(nn.Linear(in_features, out_features))
    in_features = out_features
    prev_activation = "linear"

    for i in range(1, n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 4, 128)
        function = trial.suggest_categorical(
            f"layer_function_{i}", list(activation_functions.keys())
        )

        if prev_activation == function:
            function = trial.suggest_categorical(
                f"layer_function_{i}",
                list(filter(lambda x: x != function, activation_functions.keys())),
            )
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation_functions[function]())

        prev_activation = function
        in_features = out_features

    # Add the last layer
    layers.append(nn.Linear(in_features, output_size))
    layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)
    return model


def train_test_classifier(
    trial: "optuna.Trial",
    label_name: str = "binary_severity",
    folder_path: Optional[str] = None,
    split_dataset_name: Optional[str] = "split_eclipse_72k.json",
    dataset_name: Optional[str] = "eclipse_72k",
):
    if folder_path is None:
        folder_path = f"/project/def-aloise/{os.environ['USER']}/data/"

    hdf5_file_path = Path(folder_path) / f"embeddings_chunk_v4_eclipse_layer_-1_0.hdf5"
    df = pd.read_json(folder_path / f"{dataset_name}.json")
    train_dict, val_dict, test_dict = [], [], []

    with open(folder_path / split_dataset_name, "r") as file:
        idxs = json.load(file)

    with h5py.File(hdf5_file_path, "r") as file:
        for key, value in file.items():
            severity = df[df["bug_id"] == int(key)][label_name]
            if int(key) in idxs["tr"]:
                train_dict.append(
                    {
                        "bug_id": int(key),
                        "embedding": np.array(value).tolist(),
                        label_name: int(severity.iloc[0]),
                    }
                )
            elif int(key) in idxs["val"]:
                val_dict.append(
                    {
                        "bug_id": int(key),
                        "embedding": np.array(value).tolist(),
                        label_name: int(severity.iloc[0]),
                    }
                )
            elif int(key) in idxs["test"]:
                test_dict.append(
                    {
                        "bug_id": int(key),
                        "embedding": np.array(value).tolist(),
                        label_name: int(severity.iloc[0]),
                    }
                )
            else:
                raise ValueError(f"The bug_id {key} does not exist")

    def collate_fn(data: List[dict]):
        bug_ids = [d["bug_id"] for d in data]
        inputs = [d["embedding"] for d in data]
        labels = [d[label_name] for d in data]
        return (
            torch.tensor(bug_ids, dtype=torch.float32),
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )

    # Define batch size and create a DataLoader
    batch_size = trial.suggest_categorical("batch_size", [1, 16, 32, 64])
    train_dataloader = dt.DataLoader(
        train_dict, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = dt.DataLoader(
        test_dict, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = dt.DataLoader(
        val_dict, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    input_size = len(train_dict[0]["embedding"])
    output_size = 1

    model = get_nn_classifier(
        trial=trial, input_size=input_size, output_size=output_size
    )
    # Binary Cross Entropy Loss
    pos_weight = trial.suggest_float("pos_weight", 0.1, 2.0)
    pos_weight_tensor = torch.tensor(pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # lr = learning rate

    num_epochs = trial.suggest_categorical("num_epochs", [10, 50, 100])
    total_samples = len(train_dict)
    for epoch in range(num_epochs):
        for i, (bug_ids, inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.reshape([-1, 1]))
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_samples//batch_size+1}], Loss: {loss.item():.4f}"
                )

    # Validation step
    model.eval()
    val_labels_list = []
    val_result_list = []
    with torch.no_grad():
        for bug_ids, inputs, labels in val_dataloader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            val_labels_list.extend(labels.tolist())
            result = [
                {
                    "bug_id": bug_id,
                    "binary_severity": label,
                    "prediction": prediction[0],
                }
                for bug_id, label, prediction in zip(
                    bug_ids.tolist(), labels.tolist(), predicted.tolist()
                )
            ]
            val_result_list.extend(result)
    _, val_f1, _ = compute_metrics_from_list(val_result_list, pred_field="prediction")
    _, val_class_count = np.unique(val_labels_list, return_counts=True)
    val_class_proportion = val_class_count / len(val_labels_list)
    val_weighted_avg_f1 = np.average(val_f1, weights=val_class_proportion)

    # Print validation metrics
    print(f"Validation F1 Score: {val_weighted_avg_f1}")

    # Test step
    model.eval()
    test_labels_list = []
    test_result_list = []
    with torch.no_grad():
        for bug_ids, inputs, labels in test_dataloader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            test_labels_list.extend(labels.tolist())
            result = [
                {
                    "bug_id": bug_id,
                    "binary_severity": label,
                    "prediction": prediction[0],
                }
                for bug_id, label, prediction in zip(
                    bug_ids.tolist(), labels.tolist(), predicted.tolist()
                )
            ]
            test_result_list.extend(result)
    _, f1, _ = compute_metrics_from_list(test_result_list, pred_field="prediction")
    _, class_count = np.unique(test_labels_list, return_counts=True)
    test_class_proportion = class_count / len(test_labels_list)

    weighted_avg_f1 = np.average(f1, weights=test_class_proportion)
    return weighted_avg_f1


def get_nn(path_data_folder: Optional[str] = None):
    if path_data_folder is None:
        path_data_folder = f"/project/def-aloise/{os.environ['USER']}/data/"

    study_name = "nn_classifier"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    n_jobs = 1
    study.optimize(train_test_classifier, n_trials=10, n_jobs=n_jobs)
    with open(Path(path_data_folder) / f"{study_name}results.json", "w") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value}, f)


class DataoutDict(TypedDict):
    bug_id: str
    text: str
    answer: str
    severity_pred: Union[int, float]
    binary_severity: int


@print_args
def aggr_finetune(folder_out: Path, folder_in: Path, pattern_name: str):
    folder_in = existing_path(folder_in, is_folder=True)
    folder_out.mkdir(exist_ok=True, parents=True)
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
            samples.append({**params, **metrics, "epoch": epoch})
    with open(folder_out / "metrics.json", "w") as f:
        json.dump(samples, f)


def get_embeddings_datasets(
    path_split: Path,
    path_preprocessed_data: Path,
    path_embeddings: Optional[Path] = None,
) -> Union[EmbeddingDict, DataDict]:
    with open(path_split) as fp:
        splits: SplitDict = {k: set(v) for k, v in json.load(fp).items()}  # type: ignore
    datasets = {k: [] for k in splits}
    with open(path_preprocessed_data) as fp:
        data: List[PreprocessedData] = json.load(data)  # type: ignore
    for d in data:
        for k in splits:
            if d["bug_id"] in splits[k]:
                datasets[k].append(d)
                break
    # add the embeddings
    if path_embeddings is None:
        return datasets  # type: ignore
    with h5py.File(path_embeddings) as fp:
        for k in datasets:
            for i in range(len(datasets[k])):
                bug_id = datasets[k][i]["bug_id"]
                datasets[k][i]["embedding"] = np.copy(fp[str(bug_id)])
    return datasets  # type: ignore


@print_args
def generate_train_test_split(path_data: str, folder_out: str, train_percent: float = 0.7, val_percent: float = 0.2) -> SplitDict:  # type: ignore
    """Generate a dataset with balanced training set, imbalanced validation and test sets"""
    with open(path_data) as fp:
        dataset = json.load(fp)
    path_data: Path = existing_path(path_data, is_folder=False)
    folder_out: Path = existing_path(folder_out, is_folder=True)
    assert folder_out.is_dir()
    random.seed(0)
    random.shuffle(dataset)

    positive_examples = [d for d in dataset if d["binary_severity"] == 1]
    negative_examples = [d for d in dataset if d["binary_severity"] == 0]
    num_positive_train = int(train_percent * len(positive_examples))
    num_negative_train = int(train_percent * len(negative_examples))
    num_train_per_severity = min(num_positive_train, num_negative_train)
    balanced_train_set = (
        positive_examples[:num_train_per_severity]
        + negative_examples[:num_train_per_severity]
    )
    logger.info(
        f"{num_train_per_severity=}, {len(positive_examples[:num_train_per_severity])=}, {len(negative_examples[:num_train_per_severity])=}"
    )
    assert len(balanced_train_set) == num_train_per_severity * 2
    remaining_examples = {
        1: positive_examples[num_positive_train:],
        0: negative_examples[num_negative_train:],
    }
    test_percent = 1 - train_percent - val_percent
    L = remaining_examples[1] + remaining_examples[0]
    random.shuffle(L)
    validation_set, test_set = skMsel.train_test_split(
        L,
        test_size=(test_percent / (val_percent + test_percent)),
        stratify=[e["binary_severity"] for e in L],
    )
    ids: SplitDict = {"tr": [], "val": [], "test": []}

    for k, L in zip(ids, [balanced_train_set, validation_set, test_set]):
        ids[k] = [e["bug_id"] for e in L]

    # assert sum(len(e) for e in ids.values()) == len(dataset), f"Expecting {sum(len(e) for e in ids.values())=} to be equal to {len(dataset)=}"
    with open(folder_out / f"split_{path_data.stem}.json", "w") as fp:
        json.dump(ids, fp)
    return ids


def generate_seeds(
    n_data: Optional[int] = None,
    seed_start: Optional[int] = None,
    seed_end: Optional[int] = None,
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
) -> Tuple[int, int]:
    assert (seed_start is None and seed_end is None) == (
        n_chunks is not None and interval_idx is not None and n_data is not None
    ), "Expecting either seed_start and seed_end or n_chunks and interval_idx"
    if seed_start is not None and seed_end is not None:
        if seed_end == -1:
            assert n_data is not None
            seed_end = n_data
        return (seed_start, seed_end)
    if n_chunks is not None and interval_idx is not None and n_data is not None:
        n_intervals = n_chunks
        intervals = [
            [i * (n_data // n_intervals), (i + 1) * (n_data // n_intervals)]
            for i in range(n_intervals)
        ]
        intervals[-1][1] = n_data
        [seed_start, seed_end] = intervals[interval_idx]
        return (seed_start, seed_end)
    raise Exception


@print_args
def main_compute_metrics(path_data_folder: str, input_field: str = "input", pred_field: str = "pred", id: str = "", model_name: str = default_model, token: str = default_token, n_tokens_infered_max: int = default_n_tokens_infered_max, n_tokens_show_min: int = 0, n_tokens_show_max: int = default_n_tokens_infered_max):  # type: ignore
    path_data_folder: Path = existing_path(path_data_folder, is_folder=True)
    assert_valid_token(token)
    model_name = get_literal_value(model_name)

    folder_out = path_data_folder / f"out_{pred_field}{id}"
    folder_out.mkdir(parents=True, exist_ok=True)

    tokenizer = initialize_model_inference(model_name, token, return_model=False)  # type: ignore
    fields_data = extract_fields_from_json(path_data_folder)
    conf_matrix, f1, data = compute_metrics_from_list(
        fields_data=fields_data,
        tokenizer=tokenizer,
        pred_field=pred_field,
        input_field=input_field,
        n_tokens_infered_max=n_tokens_infered_max,
        n_tokens_show_min=n_tokens_show_min,
        n_tokens_show_max=n_tokens_show_max,
    )
    compute_metrics_from_files(
        conf_matrix=conf_matrix,
        f1=f1,
        data_full=data,
        folder_out=folder_out,
        pred_field=pred_field,
        n_tokens_infered_max=n_tokens_infered_max,
        n_tokens_show_min=n_tokens_show_min,
        n_tokens_show_max=n_tokens_show_max,
    )


@print_args
def max_tokens_pipeline(folder_data: Union[str, Path], model_name: ModelName = default_model, token: str = default_token):  # type: ignore
    folder_data: Path = existing_path(folder_data, is_folder=True)
    args_dict = locals()
    (max_work, min_not_work) = get_max_tokens(
        PipelineMaxTokens(model_name=model_name, token=token),
        min_token_length=1,
        max_token_length=10000,
    )
    path_out = folder_data / "tokens_lim.json"
    L = []
    if path_out.exists():
        with open(path_out, "r") as f:
            L = json.load(f)
    L.append(
        {
            "max_work": max_work,
            "min_not_work": min_not_work,
            "model_name": model_name,
        }
    )
    for k, v in args_dict.items():
        L[-1][k] = v
    with open(path_out, "w") as f:
        json.dump(L, f, indent=2)
    logger.info(f"{max_work=},{min_not_work=}")


@print_args
def max_tokens_embeddings(folder_data: Union[str, Path], model_name: ModelName = default_model, token: str = default_token):  # type: ignore
    folder_data: Path = existing_path(folder_data, is_folder=True)
    args_dict = locals()
    (max_work, min_not_work) = get_max_tokens(
        EmbeddingsMaxTokens(model_name=model_name, token=token),
        min_token_length=1,
        max_token_length=10000,
    )
    path_out = folder_data / "tokens_lim.json"
    L = []
    if path_out.exists():
        with open(path_out, "r") as f:
            L = json.load(f)
    L.append(
        {
            "max_work": max_work,
            "min_not_work": min_not_work,
            "model_name": model_name,
        }
    )
    for k, v in args_dict.items():
        L[-1][k] = v
    with open(path_out, "w") as f:
        json.dump(L, f, indent=2)
    logger.info(f"{max_work=},{min_not_work=}")


@print_args
def max_tokens_finetuning(folder_data: Union[str, Path], qlora_alpha: int = 16, qlora_dropout: float = 0.1, model_name: ModelName = default_model, token: str = default_token, n_tokens_infered_max: int = default_n_tokens_infered_max):  # type: ignore
    folder_data: Path = existing_path(folder_data, is_folder=True)
    template_path: Path = existing_path(folder_data / "template.json", is_folder=False)
    args_dict = locals()
    path_out = folder_data / "tokens_lim.json"
    L = []
    if path_out.exists():
        with open(path_out, "r") as f:
            L = json.load(f)
    for model_name in get_args(ModelName):
        for qlora_r in [2, 8, 16, 32, 64, 128, 256]:
            (max_work, min_not_work) = get_max_tokens(
                FinetuneMaxTokens(
                    model_name=model_name,
                    token=token,
                    max_n_tokens=10000,
                    template_path=template_path,
                    lora_alpha=qlora_alpha,
                    lora_dropout=qlora_dropout,
                    lora_r=qlora_r,
                ),
                min_token_length=1,
                max_token_length=n_tokens_infered_max,
            )
            L.append(
                {
                    "max_work": max_work,
                    "min_not_work": min_not_work,
                    "qlora_r": qlora_r,
                    "model_name": model_name,
                }
            )
            for k, v in args_dict.items():
                if k not in L[-1]:
                    L[-1][k] = v
            with open(path_out, "w") as f:
                json.dump(L, f, indent=2)


if __name__ == "__main__":
    print("start")
    fire.Fire(
        {
            "split_datasets": generate_train_test_split,
            "inference": main_inference,
            "compute_metrics": main_compute_metrics,
            "generate_dataset": generate_dataset,
            "qlora_classification": main_qlora_classification,
            "max_tokens_pipeline": max_tokens_pipeline,
            "max_tokens_embeddings": max_tokens_embeddings,
            "max_tokens_finetuning": max_tokens_finetuning,
            "get_llama2_embeddings": get_llama2_embeddings,
            "merge_data_embeddings": merge_data_embeddings,
            "aggr_finetune": aggr_finetune,
            "nn_classifier": get_nn,
        }
    )
