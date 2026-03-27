from model.MLP import MLP
from model.CNN1D import CNN1D
from model.ResNet import ResNetTabular
from model.MLP_General import MLP_General
import inspect

class CoreModelFactory:
    _MODEL_DICT = {
        "mlp": MLP,
        "1dcnn": CNN1D,
        "resnet": ResNetTabular,
        "mlp_general": MLP_General,
        # Thêm model mới ở đây
    }

    @staticmethod
    def create(model_type, input_dim, class_num, **kwargs):
        key = model_type.lower()
        if key not in CoreModelFactory._MODEL_DICT:
            raise ValueError(f"Unknown model type: {model_type}")
        model_class = CoreModelFactory._MODEL_DICT[key]
        sig = inspect.signature(model_class.__init__)
        valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return model_class(input_dim, class_num, **valid_args)