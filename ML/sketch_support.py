import base64
import numpy as np

def stringToArray(b64encoded: str) -> np.ndarray:
    u = base64.b64decode(b64encoded)
    result = np.frombuffer(u, dtype=np.int16)
    return result