import pandas as pd
import numpy as np

import string
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def filter_bug_severity(dataframe, col='bug_severity'):
    # precisa filtrar a resolusao/bug_status ????
    filtered_reports = dataframe[~dataframe[col].isin(['normal', 'enhancement'])]
    selected_columns = ['_id', 'bug_id', 'description', 'bug_severity']
    return filtered_reports[selected_columns]

def create_binary_feature(dataframe, col='bug_severity'):
    def binary_feature_creator(severity):
        return 1 if severity in ['blocker', 'critical', 'major'] else 0

    bug_reports_copy = dataframe.copy()
    bug_reports_copy['binary_severity'] = bug_reports_copy[col].apply(binary_feature_creator)
    return bug_reports_copy

# check as remover without flag, and as it is in the data
def remove_code_snippets(text):
    # Remove programming code snippets enclosed in triple backticks
    code_pattern = r'```(?:[^`]+|`(?!``))*```'
    text_without_code = re.sub(code_pattern, 'CODE', text)

    # Remove programming code snippets enclosed in single backticks
    text_without_code = re.sub(r'`[^`]+`', '', text_without_code)

    return text_without_code

def remove_urls_and_codes(dataframe, col='bug_severity'):
    dataframe[col] = dataframe[col].apply(lambda text: re.sub(r'http\S+', '', text))
    dataframe[col] = dataframe[col].apply(lambda text: remove_code_snippets(text))
    return dataframe

