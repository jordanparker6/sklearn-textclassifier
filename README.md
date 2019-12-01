# sklearn-textclassifier
A production ready text classifier for models following the Scikit-Learn API.

This repository allows you to train a text classifier model and then deploy that model as a containerised Flask app served with both a gunicorn application server and Nginx webserver/reverse-proxy.

## Train

A model must be trained and pickled prior to serving. The train.py script within the train directory assists with this.

The following models are available for training:
 - LinearSVC
 - MultinomialNB
 - LogisiticRegression

### Set-Up

The train.py script requires you to set-up the train directory as follows:

<pre>
/train/
| -- input
     | -- data.csv
     | -- config.json (OPTIONAL)
     | -- hyperparameter.json (OPTIONAL)
| -- output
     | -- model.p (AFTER TRAINING)
</pre>

- data.csv is the the data file containing your text and labels.
- config.json overwrites the default training configurations.
- hyperparameter.json adjust the models default hyperparameters.
- model.p is the trained Scikit-Learn estimator object.

Default config:

```
{
    "label": "label",
    "text": "text",
    "run_eval": True,
    "model": 'LinearSVC'
}
```

- "label" is the name of the label column in data.csv.
- "text" is the name of the text column in data.csv.
- "run_eval" is a boolean switch for 5-fold cross-validation.
- "model" is the name of the model used (see available models).

### Instructions

Execute ```python train.py``` from the train directory.

## Serve

Docker and a trained model consistent with the Scikit-Learn API is required.

Envitoment Variables:
- MODEL_PATH = model.p

To launch the inference server:
 - ```git clone https://github.com/jordanparker6/sklearn-textclassifier```
 - Copy model.p to the serve directory
 - ```docker-compose build```
 - ```docker-compose up```

The app will be available at localhost:8080. This can be configured in nginx.conf.

### API EndPoints

#### Clasify

Text classification endpoint.

- Endpoint: /classify
- Method: POST
- Accepts: JSON
- Body: {
    "text": string or list of strings
}
- Returns: {
    "result": list of strings
}

#### Ping

Determine if the container is working and healthy. In this container, we declare it healthy if we can load the model successfully.

- Endpoint: /ping
- Method: GET
- Accepts: JSON
- Returns: {}
