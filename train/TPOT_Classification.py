from tpot import TPOTClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import pandas as pd

"""
Defaults:
class tpot.TPOTClassifier(generations=100, population_size=100,
                          offspring_size=None, mutation_rate=0.9,
                          crossover_rate=0.1,
                          scoring='accuracy', cv=5,
                          subsample=1.0, n_jobs=1,
                          max_time_mins=None, max_eval_time_mins=5,
                          random_state=None, config_dict=None,
                          template=None,
                          warm_start=False,
                          memory=None,
                          use_dask=False,
                          periodic_checkpoint_folder=None,
                          early_stop=None,
                          verbosity=0,
                          disable_update_check=False)
"""

data = pd.read_csv("input/data.csv").dropna()
X, y = (data["text"], data["label"])

tpot = TPOTClassifier(generations=5, warm_start=True, verbosity=2, config_dict='TPOT sparse')
pipe = make_pipeline(
        TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5),
        tpot
    )
pipe.fit(X, y)

tpot.export('output/tpot_pipeline.py')