# -*- coding: utf-8 -*-
"""baseline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10ePj7ajeztIvcy-Al6w2a9LrY1WdwCn5

# Severity prediction of bug reports

## Baseline

- Preprocess:
- Embedding using:
- Algorithms for binary classification:
- Metrics:

## Code

### Boot
"""

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

import pandas as pd
import numpy as np

import string
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer  # For text data
from sklearn.svm import SVC  # Or other classification algorithm
from sklearn.metrics import accuracy_score

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

"""### Data"""

bug_reports = pd.read_json('data/eclipse_clear.json', lines=True)
bug_reports.info()
############ fazer um script para pegar o summary

X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

"""### Pipeline"""

def filter_bug_severity(bug_reports, col='bug_severity'):
    # precisa filtrar a resolusao/bug_status ????
    filtered_reports = bug_reports[~bug_reports[col].isin(['normal', 'enhancement'])]
    selected_columns = ['_id', 'bug_id', 'description', 'bug_severity']
    return filtered_reports[selected_columns]

def create_binary_feature(bug_reports, col='bug_severity'):
    def binary_feature_creator(severity):
        return 1 if severity in ['blocker', 'critical', 'major'] else 0

    bug_reports_copy = bug_reports.copy()
    bug_reports_copy['binary_severity'] = bug_reports_copy['bug_severity'].apply(binary_feature_creator)
    return bug_reports_copy

# ver como remover sem sinalizador, e como está no dado
def remove_code_snippets(text):
    # Remove programming code snippets enclosed in triple backticks
    code_pattern = r'```(?:[^`]+|`(?!``))*```'
    text_without_code = re.sub(code_pattern, '', text)

    # Remove programming code snippets enclosed in single backticks
    text_without_code = re.sub(r'`[^`]+`', '', text_without_code)

    return text_without_code

def preprocess_text(dataframe, col='description'): #changing the inputs
    bug_reports_copy = dataframe.copy()

    # Remove URLs using regular expressions
    bug_reports_copy[col] = bug_reports_copy[col].apply(lambda text: re.sub(r'http\S+', '', text))

    # Remove programming code snippets using regular expressions
    bug_reports_copy[col] = bug_reports_copy[col].apply(remove_code_snippets)

    tokens = bug_reports_copy[col].apply(word_tokenize)
    print(tokens)

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
        # filtered_texts.append(filtered_tokens)
        filtered_texts.append(' '.join(filtered_tokens))
    print('\nfiltered_texts', filtered_texts)
    bug_reports_copy['preprocess_desc'] = filtered_texts

    return bug_reports_copy

pipeline = Pipeline([
    ('bug_severity_filter', FunctionTransformer(filter_bug_severity, kw_args={'col': 'bug_severity'})),
    ('binary_feature', FunctionTransformer(create_binary_feature, kw_args={'col': 'bug_severity'})),
    ('preprocessor', FunctionTransformer(preprocess_text)),
    # ('embedding', embedding),
    # ('classifier', classifier)
])

pipeline.fit(bug_reports)

# Get the transformed data at the filter_bug_severity step
filter_bug_severity_result = pipeline.named_steps['bug_severity_filter'].transform(bug_reports) # ver sobre a seleção das colunas
print('First step')
print('Columns:', filter_bug_severity_result.columns)
print('Return type:', type(filter_bug_severity_result))
print("Severities's kind:", filter_bug_severity_result.bug_severity.unique(), '\n')

# Get the transformed data at the binary_feature step
binary_feature_result = pipeline.named_steps['binary_feature'].transform(filter_bug_severity_result)
print('Second step')
print('Columns:', binary_feature_result.columns)
print('Return type:', type(binary_feature_result))
print("Severities's kind:", binary_feature_result.binary_severity.unique(), '\n')

# Get the transformed data at the preprocessor
#preprocessor_result = pipeline.named_steps['preprocessor'].transform(binary_feature_result)
#print('Third step')
#print('Columns:', preprocessor_result.columns)
#print('Return type:', type(preprocessor_result))
#print("Severities's kind:", preprocessor_result.head(3).preprocess_desc, '\n')

"""# Production"""

# Create the pipeline for production
production_pipeline = Pipeline([
    ('bug_severity_filter', FunctionTransformer(filter_bug_severity, kw_args={'col': 'bug_severity'})),
    ('binary_feature', FunctionTransformer(create_binary_feature, kw_args={'col': 'bug_severity'})),
    ('preprocessing', TextPreprocessor()),
    ('embedding', Word2VecEmbedding()),
    ('classifier', SVC())
])

# Fit the production pipeline on the entire dataset
production_pipeline.fit(X_text, y)  # Assuming X_text and y are already defined

# Make predictions on new, unseen data
new_data = ['New sentence 1.', 'New sentence 2.', ...]
predictions = production_pipeline.predict(new_data)
print(predictions)

"""In this production pipeline, you've removed the data splitting step and fitted the pipeline on the entire dataset. Then, you can use this pipeline to make predictions on new, unseen data by calling the predict method with the new data. This is a common approach in production when you're deploying a trained model for making predictions on real-world data."""