# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model makes a prediction if a persons salary is above or below 50k. It is extremely basic and probably just predicts one class but thats okay, its about the idea.

## Intended Use
Don't use it for anything serious, like I said, its pretty garbage.

## Training Data
Training data consists of only the numeric columns of the census.csv data. Training data is 80% randomly selected.

* age
* education-num
* capital-gain
* capital-loss
* hours-per-week
* salary <- converted to binary 0=<50k 1=>50k

## Evaluation Data
See training data section. Evaluation data is 20% randomly selected.

## Metrics
I only used the F1 score because its easy.

highest_score = 0.47

## Ethical Considerations
I am assuming the data used is fake but if it isnt well...

## Caveats and Recommendations
None
