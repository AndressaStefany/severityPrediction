import re
import abc
import string
import numpy as np
import pandas as pd
from typing import * #type: ignore
from IPython.display import display
from rich.progress import track, Progress

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, ComplementNB, MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB, GaussianNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
    
def filter_bug_severity(dataframe: pd.DataFrame, severity_col='bug_severity', severities_to_keep: Optional[Tuple[str]] = None) -> pd.DataFrame:
    """Filters the dataframe of bugs to keep only bugs within a provided severity set of possibilities
    
    # Args 
        - dataframe: pd.DataFrame, the dataframe to filter
        - severity_col: str, the name of the column containing the severity of the bug
        - severities_to_keep: Tuple[str], the severities to keep in the dataframe
        
    # Output
        - pd.DataFrame, the filtered dataframe
    """
    # precisa filtrar a resolusao/bug_status ????
    if severities_to_keep is None:
        severities_to_keep = ('normal', 'enhancement') #type: ignore
    filtered_reports = dataframe[~dataframe[severity_col].isin(severities_to_keep)] #type: ignore
    selected_columns = ['_id', 'bug_id', 'description', 'bug_severity']
    return filtered_reports[selected_columns]

def create_binary_feature(dataframe, severity_col: str ='bug_severity', high_severities_vals: Optional[Set[str]] = None) -> pd.DataFrame:
    """Creates a binary feature based on the severity of the bug
    
    # Args 
        - dataframe: pd.DataFrame, the dataframe to process
        - severity_col: str, the name of the column containing the severity of the bug
        
    # Output
        - pd.DataFrame, the dataframe with the binary feature
    """
    if high_severities_vals is None:
        high_severities_vals = {'blocker', 'critical','major'} #type: ignore

    bug_reports_copy = dataframe.copy()
    bug_reports_copy['binary_severity'] = bug_reports_copy[severity_col].apply(
        lambda severity: 1 if severity in high_severities_vals else 0
    )
    return bug_reports_copy

# check as remover without flag, and as it is in the data
def remove_code_snippets(text: str) -> str:
    """Remove programming code snippets enclosed in triple or single backticks with regex

    # Args
        - text: str, the text to process
        
    # Output
        - str, the text without code snippets
    """
    # Remove programming code snippets enclosed in triple backticks
    code_pattern_triple = r'```(?:[^`]+|`(?!``))*```'
    text_without_code = re.sub(code_pattern_triple, 'CODE', text)

    # Remove programming code snippets enclosed in single backticks
    code_pattern_single = r'`[^`]+`'
    text_without_code = re.sub(code_pattern_single, '', text_without_code)

    return text_without_code

def remove_urls_and_codes(dataframe: pd.DataFrame, col_to_process: str='description') -> pd.DataFrame:
    """Remove URLs and programming code snippets from the description column of a dataframe

    # Args
        - dataframe: pd.DataFrame, the dataframe to process
        - col: str, the name of the column containing the text to process
        
    # Output
        - pd.DataFrame, the dataframe with the URLs and programming"""
    url_pattern = r'http\S+'
    dataframe[col_to_process] = dataframe[col_to_process].apply(lambda text: re.sub(url_pattern, '', text))
    dataframe[col_to_process] = dataframe[col_to_process].apply(lambda text: remove_code_snippets(text))
    return dataframe

def preprocess_text(dataframe, col_to_process='description'):
    """Preprocess the text in the column provided of a dataframe

    # Args
        - dataframe: pd.DataFrame, the dataframe to process
        - col: str, the name of the column containing the text to process
        
    # Output
        - pd.DataFrame, the dataframe with the preprocessed text in column `preprocess_desc`
    """
    bug_reports_copy = dataframe.copy()
    
    tokens = bug_reports_copy[col_to_process].apply(word_tokenize)
    
    # Get the set of stopwords
    stop_words = set(stopwords.words('english'))

    # Define the special characters to remove
    # in the article doesn't include this step
    special_characters = set(string.punctuation)
    special_characters.add('``')
    special_characters.add("''")
    # there is also:  "n't" / checkar
    
    # Initialize the stemmer
    stemmer = PorterStemmer()
    
    # Apply stop-word removal and stemming
    filtered_texts = []
    for tokens in tokens:
        filtered_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words and token not in special_characters]
        filtered_texts.append(' '.join(filtered_tokens))
    bug_reports_copy['preprocess_desc'] = filtered_texts
    
    return bug_reports_copy

def custom_transformations() -> List[Tuple[str,FunctionTransformer]]:
    return [
        ('bug_severity_filter', FunctionTransformer(filter_bug_severity, kw_args={'severity_col': 'bug_severity'})),
        ('binary_feature', FunctionTransformer(create_binary_feature, kw_args={'severity_col': 'bug_severity'})),
        ('remove_url_and_code', FunctionTransformer(remove_urls_and_codes, kw_args={'col_to_process': 'description'})),
        ('preprocessor', FunctionTransformer(preprocess_text, kw_args={'col_to_process': 'description'}))
    ]
    
def pipeline_custom() -> Pipeline:
    """Makes the custom pipeline of the project

    # Output
        - Pipeline, the default pipeline of the project
    """
    return Pipeline(custom_transformations())
    
    
def paper_common_transformations() -> List[Tuple[str,FunctionTransformer]]:
    """Reproduces the pipeline of the article of [Lamkanfi et al. Comparing Mining Algorithms for Predicting the Severity of a Reported Bug](https://ieeexplore.ieee.org/abstract/document/5741332?casa_token=dnUxAhCPIfYAAAAA:EdGf86hHjs7WSgDOSKP9EGYDDeJzi2KWT4b9qQhqfAG931MWlbafeZaTlSC3KbwFrhJxIjpfEW4)
    
    In this pipeline, the urls and the codes of the bugs are not removed.
        
    # Output
        - List[Tuple[str,FunctionTransformer]], the common transformations to apply
    """
    def extract_x_y(df: pd.DataFrame) -> Tuple:
        return df['preprocess_desc'].values, df['binary_severity'].values
    return [
        ('bug_severity_filter', FunctionTransformer(filter_bug_severity, kw_args={'severity_col': 'bug_severity'})),
        ('binary_feature', FunctionTransformer(create_binary_feature, kw_args={'severity_col': 'bug_severity'})),
        ('preprocessor', FunctionTransformer(preprocess_text, kw_args={'col_to_process': 'description'})),
        ('extract_x_y', FunctionTransformer(extract_x_y))
    ]

def pipeline_naive_bayes(is_binomial: bool = False) -> Tuple[Pipeline,CountVectorizer]:
    """Get the pipeline of the project using the naive bayes methods

    # Args
        - is_binomial: bool, whether to use the binomial naive bayes method or not
        
    # Output
        - List, the list of the transformations applied for the naive bayes methods
    """
    vectorizer = CountVectorizer(binary=is_binomial)
    return Pipeline([
        *paper_common_transformations(),
        ('embedding', FunctionTransformer(lambda x: (vectorizer.fit_transform(x[0]).toarray(),x[1]))),
    ]), vectorizer
    
def pipeline_1NN_SVM() -> Tuple[Pipeline,CountVectorizer]:
    """Get the pipeline of the project for the 1NN and SVM methods

        
    # Output
        - List, the list of the transformations applied for the 1NN and SVM methods
    """
    vectorizer = TfidfVectorizer()
    transforms = paper_common_transformations()
    transforms.append(('embedding', FunctionTransformer(lambda x: (vectorizer.fit_transform(x[0]).toarray(),x[1]))))
    return Pipeline(transforms), vectorizer

def print_pipeline(pipeline: Pipeline, src_data):
    d = src_data
    for k,v in pipeline.named_steps.items():
        r = v.transform(d)
        print("-"*100)
        print(f"Step {k}")
        if isinstance(r, pd.DataFrame):
            print(r.info())
            display(r.head())
        else:
            print("X:",r[0][:10], type(r[0]))
            print("y:",r[1][:10], type(r[1]))
        d = r
def partial_train(X: np.ndarray,y: np.ndarray,classifier,train_indices: np.ndarray,progress: Progress,task):
    # Incremental fit to avoid memory problems
    y_train_pred = np.array([])
    y_train = np.array([])
    for tr_indices in np.array_split(train_indices, 10):
        X_train = X[tr_indices]
        y_train_partial = y[tr_indices]
        classifier.partial_fit(X_train, y_train_partial, classes=[0,1])
        
        y_train = np.concatenate([y_train,y_train_partial],axis=0)
        del X_train
        del y_train_partial
        progress.update(task,advance=1)
    for tr_indices in np.array_split(train_indices, 10):
        X_train = X[tr_indices]
        y_train_partial = y[tr_indices]
        y_train_pred = np.concatenate([y_train_pred,classifier.predict(X_train)],axis=0)
    return y_train, y_train_pred

def train(X: np.ndarray,y: np.ndarray,classifier,train_indices: np.ndarray,progress: Progress,task):
        X_train = X[train_indices]
        y_train = y[train_indices]
        classifier.fit(X_train, y_train)
        y_train_pred = classifier.predict(X_train)
        progress.update(task,advance=1)
        return y_train, y_train_pred
        
def cross_validation_with_classifier(classifier, X: np.ndarray, y: np.ndarray, n_splits: int = 5, train_fun: Optional[Callable] = None, num_rep: int = 5):
    if train_fun is None:
        train_fun = partial_train
    # make dataframe to store the seed, the classifieer used, the train method used and the train and test accuracies
    df_results = pd.DataFrame({ "seed": [], "classifier": [], "train_fun": [], "train_accuracy": [], "test_accuracy": [] , "fold_id": []})
    for seed in range(num_rep):
        # Define the k-fold cross-validation strategy
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        with Progress() as progress:
            task = progress.add_task("[red]Training...", total=n_splits*10)
            # Iterate through each fold
            for fold, (train_indices, test_indices) in enumerate(skf.split(np.zeros((len(y),)), y)):
                y_train, y_train_pred = train_fun(X,y,classifier,train_indices,progress,task)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                
                del y_train_pred
                del y_train
                X_test = X[test_indices]
                y_pred = classifier.predict(X_test)
                y_test = y[test_indices]

                test_accuracy = accuracy_score(y_test, y_pred)
                
                df_results.append({ "seed": 42, "classifier": classifier.__class__.__name__, "train_fun": train_fun.__name__, "train_accuracy": train_accuracy, "test_accuracy": test_accuracy , "fold_id": fold},ignore_index=True)

    return df_results

def save_data_to_disk(pipeline_fn: Callable, folder: Path, id: str = ""):
    bug_reports = pd.read_csv(folder / 'eclipse_filtered.csv')
    print(bug_reports.info())
    pipeline,vectorizer = pipeline_fn()
    X, y = pipeline.fit_transform(bug_reports)
    print_pipeline(pipeline,bug_reports)
    memmapped_array = np.memmap(folder / "X.npy",dtype=np.float32,mode="w+",shape=X.shape)
    with Progress() as progress:
        task = progress.add_task("[red]Loading into file...", total=X.shape[0]*X.shape[1])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                memmapped_array[i, j] = X[i,j]
                progress.update(task,advance=1)
    memmapped_array.flush()
    with open(folder / f"X_{id}.shape", "w") as shape_file:
        shape_file.write(f"({X.shape[0]},{X.shape[1]})")
    np.save(folder / f"y_{id}.npy",y)
    np.save(folder / f"X_full_{id}.npy",y)
    
def read_data_from_disk(folder: Path, id: str = "", full: bool = True):
    if full:
        X = np.load(data_path / f"y_{id}.npy")
    else:
        with open(data_path / f"X_{id}.shape", "r") as shape_file:
            shape = eval(shape_file.read().strip())
        X = np.memmap(data_path / f"X_{id}.npy", dtype='float32', mode='r',shape=shape)
    y = np.load(data_path / f"y_{id}.npy")
    return X,y

def generate_data(data_path: Path):
    save_data_to_disk(lambda :naive_bayes_classifier(is_binomial=True),data_path,id="nb_bino")
    save_data_to_disk(lambda :naive_bayes_classifier(is_binomial=False),data_path,id="nb_non_bino")
    save_data_to_disk(lambda :pipeline_1NN_SVM,data_path,id="svm_knn")
if __name__ == "__main__":
    data_path = Path("./data/")
    generate_data(data_path)
    # # Define the Naive Bayes classifier
    # classifier = BernoulliNB()
    # read_data_from_disk(data_path,id="nb_bino")
    # df = cross_validation_with_classifier(classifier,X,y)
    # out_path = data_path / "results_baselines.csv"
    # # check if out_path exists and if so get the dataframe and append the df dataframe to it and write it back into the file
    # if  out_path.exists():
    #     df_old = pd.read_csv(out_path)
    #     df = pd.concat([df_old,df])
    # df.to_csv(out_path,index=False)
    