# sklearn-textclassifier
A production ready text classifier for models following the Scikit-Learn API.

## Train

A model must be trained and saved a pickle file prior to serving. The train.py script within the train directory assists with this.

The followign models are available for training:
 - LinearSVC

### Set-Up

The train.py script requirs you to set up the following directories within the train directory:
- input: 
    - REQUIRED: a data.csv file containing your text and labels
    - OPTIONAL: a config.json file to overwrite default training configurations.
    - OPTIONAL: a hyperparameter.json file to adjust default hyperparameters.
- output:
    This is where the trained model.p file will be placed.

The train/train.py script has the following defaults:

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

To serve the trained model, Docker and a trained model consistent with the Scikit-Learn API is required.

Envitoment Variables:
- MODEL_PATH = model.p       Path to the trained model within the serve folder.

To launch the inference server:
 - ```git clone https://github.com/jordanparker6/sklearn-textclassifier```
 - Copy model.p to the serve directory
 - ```docker-compose build```
 - ```docker-compose up```

The app will be available at localhost:8080. This can be configured in nginx.conf.

### API EndPoints

#### Clasify

Text classification endpoint.

Endpoint: /classify
Method: POST
Accepts: JSON
Body: {
    "text": string or list of strings
}
Returns: {
    "result": list of strings
}

#### Ping

Determine if the container is working and healthy. In this container, we declare it healthy if we can load the model successfully.

Endpoint: /ping
Method: GET
Accepts: JSON
Returns: {}
