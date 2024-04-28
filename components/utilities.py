from pydantic import BaseModel

class Person(BaseModel):
    age: int
    workclass: str
    education_num: int
    occupation: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 27,
                    "workclass": "Private",
                    "education_num": 8,
                    "occupation": "Exec-managerial",
                    "capital_gain": 0,
                    "capital_loss": 100,
                    "hours_per_week": 12
                }
            ]
        }
    }

class TrainConfig(BaseModel):
    train_split: float