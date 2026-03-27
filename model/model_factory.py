from model.MLP_4hidden import MLP4HiddenWrapper
from model.MLP_1hidden import MLP1HiddenWrapper
from model.sklearn_wrapper import SklearnModelWrapper

class ModelFactory:
    _MODEL_DICT = {
        "mlp4hidden": MLP4HiddenWrapper,
        "mlp1hidden": MLP1HiddenWrapper,
        "random_forest": lambda input_dim, class_num: SklearnModelWrapper("random_forest"),
        "xgboost": lambda input_dim, class_num: SklearnModelWrapper("xgboost"),
    }

    @staticmethod
    def create(model_type, input_dim, class_num):
        key = model_type.lower()
        if key not in ModelFactory._MODEL_DICT:
            raise ValueError(f"Unknown downstream model type: {model_type}")
        return ModelFactory._MODEL_DICT[key](input_dim, class_num)