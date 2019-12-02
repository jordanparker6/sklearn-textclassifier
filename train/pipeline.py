import json, os
from box import Box
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

hyperparam_path = "input/hyperparams.json"
config_path = "input/config.json"

with open(config_path, 'rb') as f:
    config = json.load(f)
    args = Box(config)

if os.path.exists(hyperparam_path):
    with open(hyperparam_path, 'rb') as f:
        hyperparams = json.load(f)
else:
    hyperparams = {}

models = {
        "LinearSVC": LinearSVC,
        "MultinomialNB": MultinomialNB,
        "LogisticRegression": LogisticRegression
        }

pipe = make_pipeline(
        TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5),
        models[args.model](**hyperparams)
    )