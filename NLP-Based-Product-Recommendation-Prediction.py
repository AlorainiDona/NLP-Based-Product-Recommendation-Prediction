#!/usr/bin/env python
# coding: utf-8

# # Pipeline Project

# You will be using the provided data to create a machine learning model pipeline.
# 
# You must handle the data appropriately in your pipeline to predict whether an
# item is recommended by a customer based on their review.
# Note the data includes numerical, categorical, and text data.
# 
# You should ensure you properly train and evaluate your model.

# ## The Data

# The dataset has been anonymized and cleaned of missing values.
# 
# There are 8 features for to use to predict whether a customer recommends or does
# not recommend a product.
# The `Recommended IND` column gives whether a customer recommends the product
# where `1` is recommended and a `0` is not recommended.
# This is your model's target/

# The features can be summarized as the following:
# 
# - **Clothing ID**: Integer Categorical variable that refers to the specific piece being reviewed.
# - **Age**: Positive Integer variable of the reviewers age.
# - **Title**: String variable for the title of the review.
# - **Review Text**: String variable for the review body.
# - **Positive Feedback Count**: Positive Integer documenting the number of other customers who found this review positive.
# - **Division Name**: Categorical name of the product high level division.
# - **Department Name**: Categorical name of the product department name.
# - **Class Name**: Categorical name of the product class name.
# 
# The target:
# - **Recommended IND**: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.

# In[2]:


import re
import string
import numpy as np
import pandas as pd
import spacy
# Sklearn / Imblearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ### Loads the dataset, displays basic info, and returns the DataFrame. 

# In[3]:


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("=== Data Info ===")
    df.info()
    print("\n=== Data Head ===\n", df.head())
    return df


# ### Handles missing values by replacing nulls in 'Title' and 'Review Text' with empty strings.

# In[12]:


def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Title", "Review Text"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


# ### Combines 'Title' and 'Review Text' into a single 'Full_Review' column.

# In[5]:


def merge_text(df: pd.DataFrame) -> pd.DataFrame:
    df["Full_Review"] = df["Title"].astype(str) + " " + df["Review Text"].astype(str)
    df.drop("Title", axis=1, inplace=True)
    return df


# ### Loads the SpaCy NLP model with only essential components to improve performance.

# In[6]:


def init_spacy():
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
    return nlp


# ### Cleans and processes text by lowercasing, removing digits/punctuation, lemmatizing, and removing stopwords.

# In[7]:


def text_cleaning(text: str, nlp) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens).strip()


# ### Adds additional numerical features such as review length and sentiment indicators.

# In[8]:


def add_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df["Review Length"] = df[text_col].apply(lambda x: len(x.split()))
    df["Positive Sentiment"] = df[text_col].apply(
        lambda x: x.count("good") + x.count("love") + x.count("great")
    )
    return df


# ### Constructs a machine learning pipeline with preprocessing, SMOTE, and a RandomForest classifier.

# In[9]:


def build_pipeline() -> Pipeline:
    numeric_feats = ["Age", "Positive Feedback Count", "Review Length", "Positive Sentiment"]
    categorical_feats = ["Division Name", "Department Name", "Class Name"]
    text_feat = "Full_Review"

    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    text_transformer = TfidfVectorizer(stop_words="english", max_features=5000)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_feats),
            ("cat", cat_transformer, categorical_feats),
            ("text", text_transformer, text_feat),
        ]
    )

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced_subsample"
        ))
    ])

    return pipeline


# ### Evaluates model performance and prints key metrics and a confusion matrix.

# In[10]:


def evaluate_model(y_true, y_pred, title: str = "Model Performance"):
    print(f"\n=== {title} ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not Recommended (0)", "Recommended (1)"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# ### Complete pipeline execution: data processing, training, evaluation, and hyperparameter tuning.

# In[11]:


def main():
    df = load_data("reviews.csv")
    df = clean_nulls(df)
    df = merge_text(df)

    nlp = init_spacy()
    df["Full_Review"] = df["Full_Review"].apply(lambda x: text_cleaning(x, nlp))
    df = add_text_features(df, "Full_Review")

    # Separate X, y
    X = df.drop("Recommended IND", axis=1)
    y = df["Recommended IND"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=27, stratify=y)

    # Build pipeline and train model
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)
    evaluate_model(y_test, y_pred, "Initial Model Performance")

    # Hyperparameter Tuning
    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__min_samples_split": [2, 5],
        "preprocessor__text__max_features": [3000, 5000],
        "preprocessor__text__ngram_range": [(1,1), (1,2)],
    }

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=2, scoring="accuracy", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    evaluate_model(y_test, y_pred_tuned, "Tuned Model Performance")

if __name__ == "__main__":
    main()


# ##  Analysis
# ### - The initial model performed well with 88% accuracy.
# ### - The recall for the 'Recommended' class was very high (>95%), meaning most positive reviews were correctly identified.
# ### - After hyperparameter tuning, accuracy remained consistent, but fine-tuning parameters slightly improved balance.
# ### - The slight increase in recall suggests better handling of class imbalance.

# In[ ]:




