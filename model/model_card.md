# Model Card

For a special project we are searching for people that make above 50k yearly.

## Model Details
The model receives a list of details of an individual and predicts if that person
has a salary either above or below 50k yearly. 

## Intended Use
The model can be used to receive a prediction for an individual if their salary is above
50k yearly. It is created for an Udacity course assignment and is not intended to be used in a professional environment.

## Training Data
The model is trained on a randomly selected partition of personal data. The training partition is 80% of the labeled
dataset.

## Evaluation Data
See training data section. Evaluation data is the remaining 20% data after selecting the training data.

## Metrics
For evaluation we used the F1 score, the most recent model has a performance of 0.47 on the evaluation set.

## Ethical Considerations
The data has been anonymized, to avoid bias information pertaining the sex and ethnicity of the individual has been removed.

## Caveats and Recommendations
None
