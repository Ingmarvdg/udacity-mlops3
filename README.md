# Introduction

This is an example of a machine learning model that can be deployed using FastApi on a cloud provider of your choice.

## Prerequisites
1. a working install of Conda.
2. Python 3.8.
3. Git.

## Initial Setup
1. Clone this repository to your local machine.
2. Create a new Conda environment and install the requirements in requirements.txt
3. Happy coding!

## Upload data (optional)
The data is available inside the repository, but for good measure you can also upload the data to an S3 bucket and
access it from there. This allows you to rerun the model everytime you upload new data, instead of having to
update the repository time and time again.

## Deployment
For deployment run:

``` gunicorn -k uvicorn.workers.UvicornWorker app:app ```

in the root directory.

## API usage
The API has three endpoints:

### /metrics
Returns the metrics of the model on the test set.

### /inference
Run inference on a single sample using the model.
``` 
example_body = {
                    "age": 27,
                    "workclass": "Private",
                    "education_num": 8,
                    "occupation": "Exec-managerial",
                    "capital_gain": 0,
                    "capital_loss": 100,
                    "hours_per_week": 12,
                }
```

### /fairness
Check fairness of all values in a provided column using the model.
```
example_body = {"column": "occupation"}
```

## Model information
For the model card see model_card.md in the src/model directory.
