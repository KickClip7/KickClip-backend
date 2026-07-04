import os


def get_default_device() -> str:
    """Return default AI runtime device.

    6회차에서는 실제 torch를 dependency로 추가하지 않는다.
    그래서 torch.cuda.is_available() 같은 코드는 아직 사용하지 않고,
    환경변수 기반으로만 device 값을 정한다.

    이후 7회차에서 torch 모델을 연결할 때 torch 기반 device detection으로 교체해도 된다.
    """
    return os.getenv("KICKCLIP_AI_DEVICE", "cpu")