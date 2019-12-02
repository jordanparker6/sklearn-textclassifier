# sklearn-textclassifier
A production ready text classifier for models following the Scikit-Learn API.

## Train

A model must be trained and saved as a pickle file prior to serving. The train.py script within the train directory assists with this.

The following models are available for training:
 - LinearSVC
 - MultinomialNB
 - LogisiticRegression

### Set-Up

The train.py script requires you to set up the following directories within the train directory:
- /input
    - /data.csv - REQUIRED: The data file containing your text and labels.
    - /config.json - REQUIRED: A training configuration file.
    - /hyperparameter.json - OPTIONAL: A file to adjust default hyperparameters.
- output:
    - This is where the trained model.p file will be placed.

The config.json file has the following defaults:

```
{
    "label": "label",
    "text": "text",
    "run_eval": True,
    "model": 'LinearSVC'
}
```

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

#### Predict

The model inference endpoint.

- Endpoint: /predict
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
