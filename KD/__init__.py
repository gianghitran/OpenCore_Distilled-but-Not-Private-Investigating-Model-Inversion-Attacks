from .KD_traditional import train_student_traditionally
from .KD_TeacherOnlyKD import train_student_teacher_only

class KDFactory:
    _KD_DICT = {
        "traditional": train_student_traditionally,
        "TeacherOnlyKD": train_student_teacher_only,  # Sửa key cho đúng với config
    }

    @staticmethod
    def get_trainer(method):
        if method not in KDFactory._KD_DICT:
            raise ValueError(f"Unknown KD method: {method}")
        return KDFactory._KD_DICT[method]