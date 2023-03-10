import numpy as np


def counting_multiplicity(psi: np.ndarray, eng: np.ndarray):
    psi0 = psi[:, 0]
    eng0 = eng[0]
    for i, e in enumerate(eng[1:]):
        if e == eng0:
            psi0 = psi0 + psi[:, i + 1]
    return psi0 / (np.linalg.norm(psi0))
