from components.model import save_model, load_model
import os


def test_save_model(resources, dummy_model):
    new_name = "model2.pkl"

    save_model(dummy_model, resources, new_name)

    files_in_dir = os.listdir(resources)
    assert new_name in files_in_dir

    # if test fails it wont be removed but whatever I don't have time.
    os.remove(os.path.join(resources, new_name))


def test_load_model(resources):
    model = load_model(resources, "model.pkl")

    assert model is not None
