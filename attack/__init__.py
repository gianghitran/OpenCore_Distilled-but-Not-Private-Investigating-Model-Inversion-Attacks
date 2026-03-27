from attack.improvedAttack import ImprovedAttack
from attack.traditionalAttack import TraditionalAttack

class AttackFactory:
    _ATTACK_DICT = {
        "improved": ImprovedAttack,
        "traditional": TraditionalAttack,
    }

    @staticmethod
    def create(method, *args, **kwargs):
        if method not in AttackFactory._ATTACK_DICT:
            raise ValueError(f"Unknown attack method: {method}")
        return AttackFactory._ATTACK_DICT[method](*args, **kwargs)