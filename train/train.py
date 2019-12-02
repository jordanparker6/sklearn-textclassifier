import pandas as pd
from sklearn.model_selection import cross_validate
from box import Box
import os, pickle, json

from pipeline import pipe

config_path = "input/config.json"
data_path = "input/data.csv"
model_path = "output/model.p"

with open(config_path, 'rb') as f:
    config = json.load(f)
    args = Box(config)

def main(model, args):
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
    main(pipe, args)

