# UFC-predictor

UFC-Predictor is a machine learning module that makes predictions about UFC fights 
based solely on the physical data and historical statistics of the two fighters.
The classifier is an **ensemble** of the _RandomForestClassifier_, _LightGBM_ and _LogisticRegressor_ models.
The accuracy of the ensemble is **62.77%**: although this value seems quite far from the ideal threshold of 70%
, it is still an excellent result, as the priority was to ensure that the model was bias-free.

## Installation
The only technology required to install and run the AI module is **Docker**: this repository contains a 
docker-compose.yml file. Clone the project, open a terminal, move to the same directory of the aforementioned file, 
and run the command `docker compose up -d`. Before executing the command, ensure that ports `8000` and `8501` are not occupied.

At this point, wait a few seconds for Docker to pull the images from Docker Hub.
Once the network and the two containers have been successfully created,
open a browser and go to `http://localhost:8501/` to use the AI module.

Have fun using UFC-predictor, but don't bet too much!

## Technologies and frameworks
- **Python**: the entire project was developed using the Python 3.12 language.
- **pandas**: throughout the pipeline, the datasets were managed using pandas dataframes.
- **Scikit Learn**: this framework proved particularly useful during the training and evaluation phases. 
In particular, the [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), 
[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
and [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) models were used.
- **LightGBM**: this framework made available [LightGBM](https://lightgbm.readthedocs.io/en/stable/), a tree-based algorithm designed for faster training speed, higher efficiency and low memory usage.
- **FastAPI**: this framework was used to implement the backend REST API.
- **Streamlit**: this framework was used to implement the frontend graphical user interface.
- **Docker**: the AI module (final ensemble, REST API and GUI) has been deployed in the form of two Docker images, 
one for the [frontend](https://hub.docker.com/repository/docker/lucasalvaa/ufc-predictor-frontend)
and one for the [backend](https://hub.docker.com/repository/docker/lucasalvaa/ufc-predictor-backend).

## Context
The project was developed at the University of Salerno, Department of Computer Science, in the academic year 2025-26 
for the exam of **Machine Learning** helb by Professor Giuseppe Polese and Professor Loredana Caruccio.