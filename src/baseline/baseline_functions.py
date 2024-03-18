import re
import logging
import sys
import string
import random
from typing import Iterable
import numpy as np
from numpy import ndarray
import pandas as pd
from typing import *  # type: ignore

# from IPython.display import display
# from rich.progress import Progress
import optuna
import json
import logging
import os
import re

from scipy.sparse import csr_matrix, spmatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, ComplementNB, MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import pandas as pd
from pathlib import Path
import datetime

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import Template
import fire
import tqdm


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


# Code from src/llm/llama/main.py
DatasetName = Literal["eclipse_72k", "mozilla_200k"]
default_datasetname = "eclipse_72k"


def get_dataset_choice(dataset_choice: str) -> DatasetName:
    assert isinstance(dataset_choice, str) and dataset_choice in get_args(DatasetName)
    return dataset_choice  # type: ignore


def process(
    l: str, do_prompt: bool = True, preprompt: bool = True, add_instructions: str = ""
) -> Optional[dict]:
    """The multiprocessing function that generates the dictionnary from a line of the eclipse_clear.json file"""
    global default_severities_to_del
    global default_high_severities_vals
    data: dict = eval(l.strip())
    # filter severities
    if data["bug_severity"] in default_severities_to_del:
        return
    # binarize severities
    severity = 1 if data["bug_severity"] in default_high_severities_vals else 0
    # process descriptions
    description = data["description"]
    if isinstance(description, list):
        description = " ".join(description)
    if not isinstance(description, str):
        return
    description = description.strip()
    if description == "":
        return
    description = remove_url(description)
    if do_prompt:
        description = build_prompt(
            description, preprompt=preprompt, add_instructions=add_instructions
        )
    _id = data["_id"]
    bug_id = data["bug_id"]
    return {
        "_id": _id,
        "bug_id": bug_id,
        "severity": severity,
        "description": description,
    }


def build_few_shot(data: List[str]) -> str:
    """Build the few shot example using the data as input. It will take the first two example of severe and non severe samples as examples and remove them from data

    # Arguments
        - data: List[str], the listof lines of unprocessed json data

    # Output
        - str, the string of the additionnal instructions to add to the prompt
    """
    bugs_examples: Dict[int, dict] = {}
    for i, l in enumerate(data):
        r = process(l, do_prompt=False)
        if r is None:
            continue
        if r["severity"] not in bugs_examples and len(r["description"]) < 250:
            r["idx"] = i
            bugs_examples[r["severity"]] = r
        if 1 in bugs_examples and 0 in bugs_examples:
            break
    with open("./data/template_few_shots.txt") as f:
        t = Template(f.read())
    del data[bugs_examples[0]["idx"]]
    if bugs_examples[1]["idx"] > bugs_examples[0]["idx"]:
        bugs_examples[1]["idx"] -= 1
    else:
        del data[bugs_examples[1]["idx"]]
    return t.substitute(
        severe_descr=bugs_examples[1]["description"],
        non_severe_descr=bugs_examples[0]["description"],
    )


def build_prompt(data: str, preprompt: bool = True, add_instructions: str = ""):
    with open("./data/template.txt") as f:
        t = Template(f.read())
    preprompt_data = ""
    if preprompt:
        with open("./data/preprompt.txt") as f:
            preprompt_data = f.read().strip()
    return t.substitute(
        input=data, preprompt=preprompt_data, add_instructions=add_instructions
    )


default_severities_to_del = ("normal", "enhancement")  # type: ignore


def filter_bug_severity(
    dataframe: pd.DataFrame,
    severity_col: str = "bug_severity",
    description_col: Optional[str] = "description",
    severities_to_keep: Optional[Tuple[str]] = None,
) -> pd.DataFrame:
    """Filters the dataframe of bugs to keep only bugs within a provided severity set of possibilities

    # Args
        - dataframe: pd.DataFrame, the dataframe to filter
        - severity_col: str, the name of the column containing the severity of the bug
        - severities_to_keep: Tuple[str], the severities to keep in the dataframe

    # Output
        - pd.DataFrame, the filtered dataframe
    """
    if severities_to_keep is None:
        global default_severities_to_del
        severities_to_keep = default_severities_to_del  # type: ignore
    filtered_reports = dataframe[~dataframe[severity_col].isin(severities_to_keep)]  # type: ignore
    selected_columns = ["bug_id", description_col, "bug_severity"]
    return filtered_reports[selected_columns]


default_high_severities_vals = {"blocker", "critical", "major"}


def create_binary_feature(
    dataframe: pd.DataFrame,
    severity_col: str = "bug_severity",
    high_severities_vals: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Creates a binary feature based on the severity of the bug

    # Args
        - dataframe: pd.DataFrame, the dataframe to process
        - severity_col: str, the name of the column containing the severity of the bug

    # Output
        - pd.DataFrame, the dataframe with the binary feature
    """
    if high_severities_vals is None:
        global default_high_severities_vals
        high_severities_vals = default_high_severities_vals  # type: ignore

    bug_reports_copy = dataframe.copy()
    bug_reports_copy["binary_severity"] = bug_reports_copy[severity_col].apply(
        lambda severity: 1 if severity in high_severities_vals else 0
    )
    return bug_reports_copy


# check as remover without flag, and as it is in the data
def remove_code_snippets(text: str) -> str:
    """Remove programming code snippets enclosed in triple or single backticks with regex (does not work)

    # Args
        - text: str, the text to process

    # Output
        - str, the text without code snippets
    """
    # Remove programming code snippets enclosed in triple backticks
    code_pattern_triple = r"```(?:[^`]+|`(?!``))*```"
    text_without_code = re.sub(code_pattern_triple, "CODE", text)

    # Remove programming code snippets enclosed in single backticks
    code_pattern_single = r"`[^`]+`"
    text_without_code = re.sub(code_pattern_single, "", text_without_code)

    return text_without_code


def remove_url(text: str):
    url_pattern = r"http\S+"
    return re.sub(url_pattern, "URL", text)


def remove_urls_and_codes(
    dataframe: pd.DataFrame, col_to_process: str = "description"
) -> pd.DataFrame:
    """Remove URLs and programming code snippets from the description column of a dataframe

    # Args
        - dataframe: pd.DataFrame, the dataframe to process
        - col: str, the name of the column containing the text to process

    # Output
        - pd.DataFrame, the dataframe with the URLs and programming"""
    dataframe[col_to_process] = dataframe[col_to_process].apply(remove_url)
    dataframe[col_to_process] = dataframe[col_to_process].apply(remove_code_snippets)
    return dataframe


def preprocess_text(
    dataframe: pd.DataFrame,
    col_to_process: Optional[str] = "description",
    primary_key: Optional[str] = "bug_id",
):
    """Preprocess the text in the column provided of a dataframe

    # Args
        - dataframe: pd.DataFrame, the dataframe to process
        - col_to_process: str, the name of the column containing the text to process
        - primary_key: str, dataframe column corresponding to the primary key

    # Output
        - pd.DataFrame, the dataframe with the preprocessed text in column `description`
        and `stemmed_description`
    """
    bug_reports_copy = dataframe.copy()
    bug_reports_copy = bug_reports_copy.drop_duplicates(
        subset=[primary_key], keep="first"
    )

    # Check for empty descriptions or empty lists and remove those rows
    bug_reports_copy = bug_reports_copy[
        bug_reports_copy[col_to_process].apply(
            lambda x: isinstance(x, str) and len(x.strip()) > 0
        )
    ]

    # remove URL
    bug_reports_copy[col_to_process] = bug_reports_copy[col_to_process].apply(
        remove_url
    )

    token_list = bug_reports_copy[col_to_process].apply(word_tokenize)

    # Get the set of stopwords
    stop_words = set(stopwords.words("english"))

    # Define the special characters to remove
    # in the article doesn't include this step
    special_characters = set(string.punctuation)
    special_characters.add("``")
    special_characters.add("''")
    # there is also:  "n't" / checkar

    filtered_texts = []
    stm_filtered_texts = []

    # Initialize the stemmer
    stemmer = PorterStemmer()
    # Apply stop-word removal and stemming
    for tokens in token_list:
        stm_filtered_tokens = [
            stemmer.stem(token)
            for token in tokens
            if token.lower() not in stop_words and token not in special_characters
        ]
        stm_filtered_texts.append(" ".join(stm_filtered_tokens))

        filtered_tokens = [
            token
            for token in tokens
            if token.lower() not in stop_words and token not in special_characters
        ]
        filtered_texts.append(" ".join(filtered_tokens))

    bug_reports_copy["description"] = filtered_texts
    bug_reports_copy["stemmed_description"] = stm_filtered_texts

    # Remove rows where description is empty or a list is empty
    bug_reports_copy = bug_reports_copy[
        bug_reports_copy["description"].apply(
            lambda x: isinstance(x, str) and len(x.strip()) > 0
        )
    ]
    cols = [
        "bug_id",
        "bug_severity",
        "binary_severity",
        "description",
        "stemmed_description",
    ]
    return bug_reports_copy[cols]


def custom_transformations() -> List[Tuple[str, FunctionTransformer]]:
    return [
        (
            "bug_severity_filter",
            FunctionTransformer(
                filter_bug_severity, kw_args={"severity_col": "bug_severity"}
            ),
        ),
        (
            "binary_feature",
            FunctionTransformer(
                create_binary_feature, kw_args={"severity_col": "bug_severity"}
            ),
        ),
        (
            "remove_url_and_code",
            FunctionTransformer(
                remove_urls_and_codes, kw_args={"col_to_process": "description"}
            ),
        ),
        (
            "preprocessor",
            FunctionTransformer(
                preprocess_text, kw_args={"col_to_process": "description"}
            ),
        ),
    ]


def pipeline_custom() -> Pipeline:
    """Makes the custom pipeline of the project

    # Output
        - Pipeline, the default pipeline of the project
    """
    return Pipeline(custom_transformations())


def old_extract_x_y(df: pd.DataFrame) -> Tuple:
    return df["preprocess_desc"].values, df["binary_severity"].values


def full_paper_common_transformations() -> List[Tuple[str, FunctionTransformer]]:
    """Reproduces the pipeline of the article of [Lamkanfi et al. Comparing Mining Algorithms for Predicting the Severity of a Reported Bug](https://ieeexplore.ieee.org/abstract/document/5741332?casa_token=dnUxAhCPIfYAAAAA:EdGf86hHjs7WSgDOSKP9EGYDDeJzi2KWT4b9qQhqfAG931MWlbafeZaTlSC3KbwFrhJxIjpfEW4)

    In this pipeline, the urls and the codes of the bugs are not removed.

    # Output
        - List[Tuple[str,FunctionTransformer]], the common transformations to apply
    """
    return [
        (
            "bug_severity_filter",
            FunctionTransformer(
                filter_bug_severity, kw_args={"severity_col": "bug_severity"}
            ),
        ),
        (
            "binary_feature",
            FunctionTransformer(
                create_binary_feature, kw_args={"severity_col": "bug_severity"}
            ),
        ),
        (
            "preprocessor",
            FunctionTransformer(
                preprocess_text, kw_args={"col_to_process": "description"}
            ),
        ),
        ("extract_x_y", FunctionTransformer(old_extract_x_y)),
    ]


def extract_x_y(df: pd.DataFrame) -> Tuple:
    return df["stemmed_description"].values, df["binary_severity"].values


def paper_common_transformations() -> List[Tuple[str, FunctionTransformer]]:
    """Reproduces the pipeline of the article of [Lamkanfi et al. Comparing Mining Algorithms for Predicting the Severity of a Reported Bug](https://ieeexplore.ieee.org/abstract/document/5741332?casa_token=dnUxAhCPIfYAAAAA:EdGf86hHjs7WSgDOSKP9EGYDDeJzi2KWT4b9qQhqfAG931MWlbafeZaTlSC3KbwFrhJxIjpfEW4)

    In this pipeline, the urls and the codes of the bugs are not removed. We assume that all preprocessing and stemming has been already done and that we have binary_severity and stemmed_description


    # Output
        - List[Tuple[str,FunctionTransformer]], the common transformations to apply
    """
    return [("extract_x_y", FunctionTransformer(extract_x_y))]


class CustomTfidfVectorizer(TfidfVectorizer):
    def fit_transform(self, data: Tuple, *args, **kwargs) -> tuple:
        return super().fit_transform(data[0], *args, **kwargs).toarray(), data[1]  # type: ignore

    def fit(self, data: Tuple, *args, **kwargs):
        return super().fit(data[0], *args, **kwargs)

    def transform(self, data: Tuple, *args, **kwargs) -> tuple:
        return super().transform(data[0], *args, **kwargs).toarray(), data[1]


class CustomCountVectorizer(CountVectorizer):
    def fit_transform(self, data: Tuple, *args, **kwargs) -> tuple:
        return super().fit_transform(data[0], *args, **kwargs).toarray(), data[1]  # type: ignore

    def fit(self, data: Tuple, *args, **kwargs):
        return super().fit(data[0], *args, **kwargs)

    def transform(self, data: Tuple, *args, **kwargs) -> tuple:
        return super().transform(data[0], *args, **kwargs).toarray(), data[1]


class CustomMinMaxScaler(MinMaxScaler):
    def fit_transform(self, data: Tuple, *args, **kwargs) -> tuple:
        return super().fit_transform(data[0], *args, **kwargs), data[1]

    def fit(self, data: Tuple, *args, **kwargs):
        return super().fit(data[0], *args, **kwargs)

    def transform(self, data: Tuple, *args, **kwargs) -> tuple:
        return super().transform(data[0], *args, **kwargs), data[1]


def pipeline_naive_bayes(is_binomial: bool = False) -> Tuple[Pipeline, CountVectorizer]:
    """Get the pipeline of the project using the naive bayes methods

    # Args
        - is_binomial: bool, whether to use the binomial naive bayes method or not

    # Output
        - List, the list of the transformations applied for the naive bayes methods
    """
    vectorizer = CustomCountVectorizer(binary=is_binomial)
    transforms = paper_common_transformations()
    transforms.extend(
        [
            (
                "embedding",
                vectorizer,  # type: ignore
            ),
            (
                "normalization",
                CustomMinMaxScaler(),  # type: ignore
            ),
        ]
    )
    return Pipeline(transforms), vectorizer


def pipeline_1NN_SVM() -> Tuple[Pipeline, CountVectorizer]:
    """Get the pipeline of the project for the 1NN and SVM methods


    # Output
        - List, the list of the transformations applied for the 1NN and SVM methods
    """
    vectorizer = CustomCountVectorizer()
    transforms = paper_common_transformations()
    transforms.extend(
        [
            (
                "embedding",
                vectorizer,  # type: ignore
            ),
            (
                "normalization",
                CustomMinMaxScaler(),  # type: ignore
            ),
        ]
    )
    return Pipeline(transforms), vectorizer


def print_pipeline(pipeline: Pipeline, src_data):
    d = src_data
    for k, v in pipeline.named_steps.items():
        r = v.transform(d)
        print("-" * 100)
        print(f"Step {k}")
        if isinstance(r, pd.DataFrame):
            print(r.info())
            display(r.head())
        else:
            print("X:", r[0][:10], type(r[0]))
            print("y:", r[1][:10], type(r[1]))
        d = r


def partial_train(X: np.ndarray, y: np.ndarray, classifier, train_indices: np.ndarray):
    # Incremental fit to avoid memory problems
    y_train_pred = np.array([])
    y_train = np.array([])
    for tr_indices in np.array_split(train_indices, 10):
        X_train = X[tr_indices]
        y_train_partial = y[tr_indices]
        classifier.partial_fit(X_train, y_train_partial, classes=[0, 1])

        y_train = np.concatenate([y_train, y_train_partial], axis=0)
        del X_train
        del y_train_partial
    for tr_indices in np.array_split(train_indices, 10):
        X_train = X[tr_indices]
        y_train_partial = y[tr_indices]
        y_train_pred = np.concatenate(
            [y_train_pred, classifier.predict(X_train)], axis=0
        )
    return y_train, y_train_pred


class PredictionError(Exception):
    pass


def train(X_tr: np.ndarray, y_tr: np.ndarray, classifier):
    classifier.fit(X_tr, y_tr)
    y_train_pred = classifier.predict_proba(X_tr)[:, 1]
    nan_X = np.isnan(X_tr.flatten())
    nan_y = np.isnan(y_tr.flatten())
    nan_y_pred = np.isnan(y_train_pred.flatten())
    if True in nan_X:
        raise PredictionError("There are nan after float16 in X_tr")
    if True in nan_y:
        raise PredictionError("There are nan after float16 in y_tr")
    if True in nan_y_pred:
        raise PredictionError(
            f"There are nan after float16 in y_pred {classifier.__class__.__name__}"
        )
    return y_tr, y_train_pred


def cross_validation_with_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    train_fun: Optional[Callable] = None,
    num_rep: int = 5,
    **classifier_args,
):
    if train_fun is None:
        train_fun = partial_train
    classifier = get_classifier(**classifier_args)
    # make dataframe to store the seed, the classifieer used, the train method used and the train and test accuracies
    df_results = []
    for seed in range(num_rep):
        # Define the k-fold cross-validation strategy
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # Iterate through each fold
        # print(f"Seed {seed}: ",end="")
        for fold, (train_indices, test_indices) in enumerate(
            skf.split(np.zeros((len(y),)), y)
        ):
            y_train, y_train_pred = train_fun(X, y, classifier, train_indices)
            train_accuracy = accuracy_score(
                y_train, y_train_pred.round(decimals=0).astype(int)
            )

            del y_train_pred
            del y_train
            X_test = X[test_indices][:1000]
            y_pred = classifier.predict(X_test)
            y_test = y[test_indices][:1000]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            test_accuracy = accuracy_score(y_test, y_pred.round(decimals=0).astype(int))
            df_results.append(
                {
                    "seed": seed,
                    "classifier": classifier.__class__.__name__,
                    "train_fun": train_fun.__name__,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "fold_id": fold,
                    "roc_auc": roc_auc,
                    **classifier_args,
                }
            )
            # print(f"f{fold}",end=" ")
        # print()

    return pd.DataFrame(df_results)


def train_valid_test(
    logger: logging.Logger,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_fun: Optional[Callable] = None,
    num_rep: int = 5,
    test: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    **classifier_args,
):
    if train_fun is None:
        train_fun = partial_train
    classifier = get_classifier(**classifier_args)
    df_results = []
    # logger.info(f"Train data loaded {X_tr.shape=} {y_tr.shape=}")
    y_train, y_train_pred = train_fun(X_tr, y_tr, classifier)
    train_accuracy = accuracy_score(y_train, y_train_pred.round(decimals=0).astype(int))

    del y_train_pred
    del y_train
    y_val_pred = classifier.predict_proba(X_val)[:, 1]
    nan_X = np.isnan(X_val.flatten())
    nan_y = np.isnan(y_val.flatten())
    nan_y_pred = np.isnan(y_val_pred.flatten())
    if True in nan_X:
        raise PredictionError("There are nan after float16 in X_val")
    if True in nan_y:
        raise PredictionError("There are nan after float16 in y_val")
    if True in nan_y_pred:
        raise PredictionError("There are nan after float16 in y_pred_val")
    fpr, tpr, thresholds_val = roc_curve(y_val, y_val_pred)
    auc_val = auc(fpr, tpr)

    auc_test = -1
    y_test_pred = []
    y_test = []
    thresholds_test = np.ndarray([])
    if test is not None:
        X_test, y_test = test
        y_test_pred = classifier.predict_proba(X_test)[:, 1]
        nan_X = np.isnan(X_test.flatten())
        nan_y = np.isnan(y_test.flatten())
        nan_y_pred = np.isnan(y_test_pred.flatten())
        if True in nan_X:
            raise PredictionError("There are nan after float16 in X_test")
        if True in nan_y:
            raise PredictionError("There are nan after float16 in y_test")
        if True in nan_y_pred:
            raise PredictionError("There are nan after float16 in y_pred_test")
        fpr, tpr, thresholds_test = roc_curve(y_val, y_val_pred)
        auc_test = auc(fpr, tpr)
    df_results.append(
        {
            "seed": 0,
            "thresholds_val": thresholds_val.tolist(),
            "classifier": classifier.__class__.__name__,
            "train_fun": train_fun.__name__,
            "train_accuracy": train_accuracy,
            "fold_id": 0,
            "auc_val": auc_val,
            **classifier_args,
            "binary_severity_val": y_val,
            "y_val_pred": y_val_pred,
            "binary_severity_test": y_test,
            "y_test_pred": y_test_pred,
            "auc_test": auc_test,
            "thresholds_test": thresholds_test.tolist(),
        }
    )

    return pd.DataFrame(df_results)


def save_data_to_disk(
    split: dict,
    num_samples: Tuple[int],
    pipeline_fn: Callable,
    folder: Path,
    id: str = "",
    do_print: bool = False,
    dataset_choice: DatasetName = default_datasetname,
):
    dataset_choice = get_dataset_choice(dataset_choice)
    bug_reports = pd.read_json(folder / f"{dataset_choice}.json")
    bug_reports.loc[:, "description"] = bug_reports.loc[:, "stemmed_description"]
    bug_reports.loc[:, "preprocess_desc"] = bug_reports.loc[:, "stemmed_description"]

    if do_print:
        print(bug_reports.info())
    df = pd.read_json(folder / f"{dataset_choice}.json")
    for n_samples in num_samples:
        pipeline, vectorizer = pipeline_fn()
        dataset_type = "tr"
        if n_samples == -1:
            n_samples = len(split[dataset_type])
        if n_samples == -1:
            bug_id_tr = split["tr"]
            bug_id_val = split["val"]
        else:
            bug_id_tr = split["tr"][:n_samples]
            bug_id_val = split["val"][:n_samples]
        tr_samples = df[df["bug_id"].isin(bug_id_tr)].copy()
        val_samples = df[df["bug_id"].isin(bug_id_val)].copy()
        pipeline.fit(tr_samples)
        X_tr, y_tr = pipeline.transform(tr_samples)
        X_val, y_val = pipeline.transform(val_samples)
        X_all_val, y_all_val = pipeline.transform(df[df["bug_id"].isin(split["val"])].copy())
        X_all_test, y_all_test = pipeline.transform(df[df["bug_id"].isin(split["test"])].copy())
        assert (
            X_val.shape[1] == X_tr.shape[1]
        ), f"Mismatch in dimension {X_val.shape=} {X_tr.shape=}"

        size = n_samples
        if n_samples == -1:
            size = len(split["tr"])
        # np.save(
        #     folder / f"X_full_tr_{id}_{dataset_choice}_{size}_samples.npy",
        #     X_tr,
        # )
        # np.save(
        #     folder / f"y_tr_{id}_{dataset_choice}_{size}_samples.npy",
        #     y_tr,
        # )
        # if n_samples == -1:
        #     size = len(split["val"])
        # np.save(
        #     folder / f"X_full_val_{id}_{dataset_choice}_{size}_samples.npy",
        #     X_val,
        # )
        # np.save(
        #     folder / f"y_val_{id}_{dataset_choice}_{size}_samples.npy",
        #     y_val,
        # )
        # np.save(
        #     folder / f"X_full_all_val_{id}_{dataset_choice}_{size}_samples.npy",
        #     X_all_val,
        # )
        # np.save(
        #     folder / f"y_all_val_{id}_{dataset_choice}_{size}_samples.npy",
        #     y_all_val,
        # )
        np.save(
            folder / f"X_full_all_test_{id}_{dataset_choice}_{size}_samples.npy",
            X_all_test,
        )
        np.save(
            folder / f"y_all_test_{id}_{dataset_choice}_{size}_samples.npy",
            y_all_test,
        )

get_num_samples = lambda x: int(re.findall("([0-9]+)_samples", x.stem)[0])


def read_data_from_disk(
    folder: Path, id: str = "", full: bool = True, full_valid: bool = False
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Optional[Tuple[np.ndarray, np.ndarray]],
]:
    """id must contain the _num_samples"""
    if full:
        X_tr = np.load(folder / f"X_full_tr_{id}.npy")
        y_tr = np.load(folder / f"y_tr_{id}.npy")
        test = None
        if full_valid:
            X_val = np.load(folder / f"X_full_all_val_{id}.npy")
            y_val = np.load(folder / f"y_all_val_{id}.npy")
            X_test = np.load(folder / f"X_full_all_test_{id}.npy")
            y_test = np.load(folder / f"y_all_test_{id}.npy")
            test = X_test, y_test
        else:
            X_val = np.load(folder / f"X_full_val_{id}.npy")
            y_val = np.load(folder / f"y_val_{id}.npy")
        assert X_tr.shape[1] == X_val.shape[1], f"Expecting to have the same number of features for tr {X_tr.shape[1]} and val {X_val.shape[1]} for {full_valid=}"
        return (X_tr, y_tr), (X_val, y_val), test
    else:
        raise NotImplemented


def generate_data(
    data_path: Path, num_samples: Tuple[int], dataset: DatasetName, split: dict
):
    random.seed(0)
    random.shuffle(split["tr"])
    save_data_to_disk(
        split,
        num_samples,
        lambda: pipeline_naive_bayes(is_binomial=True),
        data_path,
        id="nb_bino",
        dataset_choice=dataset,
    )
    save_data_to_disk(
        split,
        num_samples,
        lambda: pipeline_naive_bayes(is_binomial=False),
        data_path,
        id="nb_non_bino",
        dataset_choice=dataset,
    )
    save_data_to_disk(
        split,
        num_samples,
        pipeline_1NN_SVM,
        data_path,
        id="svm_knn",
        dataset_choice=dataset,
    )


def get_classifier(classifier_name: str, **kwargs):
    if classifier_name == "BernoulliNB":
        return BernoulliNB(**kwargs)
    elif classifier_name == "MultinomialNB":
        return MultinomialNB(**kwargs)
    elif classifier_name == "GaussianNB":
        return GaussianNB(**kwargs)
    elif classifier_name == "ComplementNB":
        return ComplementNB(**kwargs)
    elif classifier_name == "SVC":
        if "C" not in kwargs:
            kwargs["C"] = 100.0
        if "kernel" not in kwargs:
            kwargs["kernel"] = "rbf"
        if "probability" not in kwargs:
            kwargs["probability"] = True
        if "gamma" not in kwargs:
            kwargs["gamma"] = 0.001
        return SVC(**kwargs)
    elif classifier_name == "KNeighborsClassifier":
        if "n_neighbors" not in kwargs:
            kwargs["n_neighbors"] = 5
        return KNeighborsClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown classifier {classifier_name}")


ClassifierName = Literal[
    "BernoulliNB",
    "MultinomialNB",
    "GaussianNB",
    "ComplementNB",
    "SVC",
    "KNeighborsClassifier",
]


def run_optuna(
    trial: optuna.Trial,
    dataset: str,
    models: Optional[List[ClassifierName]] = None,
    num_rep: int = 5,
    num_samples: int = -1,
    save_study_name: str = "",
):
    folder = Path("./data/")
    if models is None:
        models = ["BernoulliNB", "MultinomialNB", "GaussianNB", "ComplementNB"]
    classifier_name = trial.suggest_categorical("classifier_name", models)
    kwargs = {}
    if classifier_name == "BernoulliNB":
        id = f"nb_bino_{dataset}_{num_samples}_samples"
        full = True
    elif classifier_name in ["MultinomialNB", "GaussianNB", "ComplementNB"]:
        id = f"nb_non_bino_{dataset}_{num_samples}_samples"
        full = True
    elif classifier_name in ["SVC", "KNeighborsClassifier"]:
        id = f"svm_knn_{dataset}_{num_samples}_samples"
        full = True
    else:
        raise ValueError(f"classifier_name {classifier_name} is not a possible name")
    kwargs["classifier_name"] = classifier_name
    if classifier_name in ["BernoulliNB", "MultinomialNB", "ComplementNB"]:
        kwargs["alpha"] = trial.suggest_float("alpha", 1e-10, 1, log=True)
        kwargs["fit_prior"] = trial.suggest_categorical("fit_prior", [True, False])
    if classifier_name in ["BernoulliNB", "MultinomialNB", "GaussianNB", "ComplementNB"]:
        class_prior_non_severe = trial.suggest_float("class_prior", 0.0, 1.0)
        kwargs["class_prior"] = np.array([class_prior_non_severe, 1-class_prior_non_severe])
    if classifier_name in ["BernoulliNB"]:
        kwargs["binarize"] = trial.suggest_float("binarize", 0.0, 1.0)
    if classifier_name == "ComplementNB":
        kwargs["norm"] = trial.suggest_categorical("norm", [False, True])
    if classifier_name == "GaussianNB":
        kwargs["var_smoothing"] = trial.suggest_float(
            "var_smoothing", 1e-14, 1, log=True
        )
    if classifier_name == "SVC":
        kernel = trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        )
        is_max_iter = trial.suggest_categorical("is_max_iter", [True, False])
        if not is_max_iter:
            max_iter = -1
        else:
            max_iter = trial.suggest_int("max_iter", 1, 1000)
        decision_function_shape = trial.suggest_categorical(
            "decision_function_shape", ["ovo", "ovr"]
        )
        kwargs = {
            "classifier_name": "SVC",
            "C": trial.suggest_float("C", 1e-5, 1e5, log=True),
            "kernel": kernel,
            "degree": trial.suggest_int("degree", 1, 5) if kernel == "poly" else 3,
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
            if kernel in ["rbf", "poly", "sigmoid"]
            else "scale",
            "coef0": trial.suggest_float("coef0", -1.0, 1.0)
            if kernel in ["poly", "sigmoid"]
            else 0,
            "shrinking": trial.suggest_categorical("shrinking", [True, False]),
            "probability": True,
            "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
            "max_iter": max_iter,
            "decision_function_shape": decision_function_shape,
            "break_ties": trial.suggest_categorical("break_ties", [True, False])
            if decision_function_shape != "ovo"
            else False,
        }
    if classifier_name == "KNeighborsClassifier":
        kwargs = {
            "classifier_name": "KNeighborsClassifier",
            "n_neighbors": trial.suggest_categorical(
                "n_neighbors", [1, 5, 7]
            ),  # Number of neighbors
            "weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),  # Weight function
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),  # Algorithm not "ball_tree", "kd_tree" because sparse input
            "leaf_size": trial.suggest_categorical(
                "leaf_size", [10, 20, 30, 40, 50]
            ),  # Leaf size
            "p": trial.suggest_categorical(
                "p", [1, 2, 3]
            ),  # Power parameter for the Minkowski metric (1 for Manhattan, 2 for Euclidean, 3 Minkwski)
        }
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )  # Define the log message format
    (X_tr, y_tr), (X_val, y_val), test = read_data_from_disk(
        folder, id, full=full, full_valid=save_study_name != ""
    )
    try:
        df = train_valid_test(
            X_tr=X_tr,
            y_tr=y_tr,
            X_val=X_val,
            y_val=y_val,
            test=test,
            train_fun=train,
            **kwargs,
            num_rep=num_rep,
            logger=logger,
        )
    except PredictionError:
        raise optuna.exceptions.TrialPruned()
    if save_study_name != "":
        path = f"./{kwargs['classifier_name']}_{id}_{0}.json"
        df.to_json(path, orient="records")
    del X_tr
    del y_tr
    del X_val
    del y_val
    del test
    value = df["auc_val"].mean()
    return value


@print_args
def hyperparameter_search(
    id: str,
    dataset: str,
    models: Optional[List[ClassifierName]] = None,
    n_jobs: int = 4,
    num_rep: int = 5,
    num_samples: int = -1,
    id_job: str = "",
):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    id += "_" + dataset
    # prepare the data
    folder_out = Path("data/")
    with open(folder_out / f"split_{dataset}.json") as fp:
        split = json.load(fp)
    if num_samples != -1:
        split["tr"] = split["tr"][: min(num_samples, len(split["tr"]))]
        split["val"] = split["val"][: min(num_samples, len(split["val"]))]
    id += f"_{num_samples}_samples"
    print(f"Using id {id}")
    study_name = f"study-{id}-{id_job}"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    sampler = optuna.samplers.RandomSampler(seed=0)
    storage = optuna.storages.RDBStorage(
        storage_name, engine_kwargs={"connect_args": {"timeout": 20.0}}
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )
    study.optimize(
        lambda trial: run_optuna(
            trial,
            models=models,
            num_rep=num_rep,
            dataset=dataset,
            num_samples=num_samples,
        ),
        n_trials=50,
        n_jobs=n_jobs,
    )
    with open(f"data/study-{id}-{id_job}-best.json", "w") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value}, f)
    run_optuna(optuna.trial.FixedTrial(params=study.best_params), num_samples=num_samples, dataset=dataset, models=models, save_study_name=study_name)  # type: ignore


@print_args
def generate_dataset(dataset: DatasetName, num_samples: Tuple[int]):  # type: ignore
    assert isinstance(dataset, str) and dataset in get_args(DatasetName)
    data_path = Path("./data/")
    with open(data_path / f"split_{dataset}.json") as fp:
        split = json.load(fp)
    generate_data(data_path, dataset=dataset, split=split, num_samples=num_samples)


AlgorithmName = Literal["bayesian", "SVC", "KNN"]


def launch_search(
    algorithm: AlgorithmName,
    dataset: DatasetName,
    num_jobs: int = 4,
    num_samples: int = -1,
    id_job: str = "",
    num_rep: int = 5,
):
    assert isinstance(algorithm, str) and algorithm in get_args(AlgorithmName)
    assert isinstance(dataset, str) and dataset in get_args(DatasetName)
    if algorithm == "bayesian":
        hyperparameter_search(
            "bayesian-networks",
            dataset,
            ["BernoulliNB", "ComplementNB", "GaussianNB", "MultinomialNB"],
            n_jobs=num_jobs,
            num_samples=num_samples,
            id_job=id_job,
            num_rep=num_rep,
        )
    elif algorithm == "SVC":
        hyperparameter_search(
            "svc",
            dataset,
            ["SVC"],
            n_jobs=num_jobs,
            num_samples=num_samples,
            id_job=id_job,
            num_rep=num_rep,
        )
    elif algorithm == "KNN":
        hyperparameter_search(
            "knn",
            dataset,
            ["KNeighborsClassifier"],
            n_jobs=num_jobs,
            num_samples=num_samples,
            id_job=id_job,
            num_rep=num_rep,
        )
    else:
        raise Exception

@print_args
def reproduce_best_studies(
    folder_in: Optional[Path] = None,
    pattern: str = "*.db",
    folder_out: Optional[Path] = None,
):
    if folder_in is None:
        folder_in = Path(".")
    folder_in = Path(folder_in)
    if folder_out is None:
        folder_out = Path(".")
    folder_out = Path(folder_out).resolve()
    folder_in = Path(folder_in).resolve()
    folder_out.mkdir(exist_ok=True, parents=True)
    print(locals())
    files = sorted(list(folder_in.rglob(pattern)), key=get_num_samples)
    for f in tqdm.tqdm(files, desc="Processing"):
        stem = f.stem
        print(stem)
        dataset_choice = "eclipse_72k" if "eclipse_72k" in stem else "mozilla_200k"
        models: List[ClassifierName] = []
        if "svc" in stem:
            models = ["SVC"]
        elif "bayesian-networks" in stem:
            models = ["BernoulliNB", "ComplementNB", "GaussianNB", "MultinomialNB"]
        elif "knn" in stem:
            models = ["KNeighborsClassifier"]
        else:
            raise Exception
        old_folder = Path(".").resolve()
        os.chdir(folder_in.as_posix())
        storage = optuna.storages.RDBStorage(
            f"sqlite:///{stem}.db", engine_kwargs={"connect_args": {"timeout": 20.0}}
        )
        study = optuna.load_study(study_name=stem, storage=storage)
        os.chdir(old_folder.as_posix())
        best_trial = study.best_trial
        params = best_trial.params
        num_samples = get_num_samples(f)
        run_optuna(optuna.trial.FixedTrial(params=params), num_samples=num_samples, dataset=dataset_choice, models=models, save_study_name=stem)  # type: ignore


if __name__ == "__main__":
    fire.Fire(
        {
            "generate_dataset": generate_dataset,
            "launch_search": launch_search,
            "reproduce_best_studies": reproduce_best_studies,
        }
    )
# TODO: reproduce best at the end of search
