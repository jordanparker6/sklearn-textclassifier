import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from box import Box
import os, pickle, json

hyperparam_path = "input/hyperparams.json"
config_path = "input/config.json"
data_path = "input/data.csv"
model_path = "output/model.p"

defaults={
    "label": "label",
    "text": "text",
    "run_eval": True,
    "model": 'LinearSVC'
}

if os.path.exists(config_path):
    with open(config_path, 'rb') as f:
        config = json.load(f)
else:
    config = defaults

if os.path.exists(hyperparam_path):
    with open(hyperparam_path, 'rb') as f:
        hyperparams = json.load(f)
else:
    hyperparams = {}

args = Box(config)


def main(args, hyperparams):
    models = {
        "LinearSVC": LinearSVC(**hyperparams),
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression()
        }

    model = make_pipeline(
        TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5),
        models[args.model]
    )

    print("Loading data...")
    data = pd.read_csv(data_path).dropna()
    X, y = (data[args.text], data[args.label])

    print("Training model...")
    cv = 5 if args.run_eval else 0
    result = cross_validate(model, X, y,
                            cv=cv,
                            scoring='accuracy',
                            n_jobs=-1,
                            verbose=1,
                            return_train_score=True,
                            return_estimator=True)

    print("Model trained!")
    print("Average training score: ", result["train_score"].mean())
    print("Average train time score: ", result["fit_time"].mean())
    print("Average evaluation score: ", result["test_score"].mean())

    model = result["estimator"][0]
    print('Saving model...')
    with open(os.path.join(model_path), 'wb') as f:
        pickle.dump(model, f)
    print('Model saved!')

if __name__ == "__main__":
    main(args, hyperparams)

