## Deploying ML Models with FastAPI and Docker

```
project-directory/
├── Dockerfile # Docker configuration
├── README.md
├── app
│   ├── __init__.py # Empty file
│   └── main.py     # FastAPI logic
├── model
│   └── linear_regression_model.pkl # Saved model (after running model_training.py)
├── model_training.py # Model training code
├── poetry.lock       # poetry lock file 
├── pyproject.toml    # poetry toml file
└── requirements.txt  # Python dependencies
```
## Installation

1. Clone the repository:

```bash
git clone https://github.com/csjp/model_deployment.git
cd model_deployment
```

2. Install Poetry (if not already installed):

```bash
curl -sSL https://install.python-poetry.org | python -
```

3. Install dependencies:

```bash
poetry install
```

Run the script to train the model and save it:

```
$ poetyr run python model_training.py
```

You should be able to find the .pkl file (`linear_regression_model.pkl`) in the `model/` directory.

Use FastAPI to build an API to serve model predictions and containerize it using Docker.

### Building the Docker Image 

Build the Docker image by running the following `docker build` command:

```
$ docker build -t house-price-prediction-api .
```

Next run the Docker container:

```
$ docker run -d -p 80:80 house-price-prediction-api
```

Your API should now be running and accessible at http://127.0.0.1:80.

You can use curl or Postman to test the /predict endpoint by sending a POST request. Here’s an example request:
```
curl -X 'POST' \
  'http://127.0.0.1:80/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "MedInc": 3.5,
  "AveRooms": 5.0,
  "AveOccup": 2.0
}'
```

### Tagging and Pushing the Image to Docker Hub

First, login to Docker Hub:

```
$ docker login
```

Tag the Docker image:

```
$ docker tag house-price-prediction-api your_username/house-price-prediction-api:v1
```

Push the image to Docker Hub:

```
$ docker push your_username/house-price-prediction-api:v1
```

Other developers can now pull and run the image like so: 

```
$ docker pull yoyomaper/house-price-prediction-api:v1
$ docker run -d -p 80:80 yoyomaper/house-price-prediction-api:v1
```

Your API should now be running and accessible at http://127.0.0.1:80.

You can use curl or Postman to test the /predict endpoint by sending a POST request. Here’s an example request:
```
curl -X 'POST' \
  'http://127.0.0.1:80/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "MedInc": 3.5,
  "AveRooms": 5.0,
  "AveOccup": 2.0
}'
```




