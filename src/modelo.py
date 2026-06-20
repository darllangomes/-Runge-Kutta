import numpy as np


def f(t, y, R, L, C, fonte):
    """Lado direito do sistema de 1a ordem do circuito RLC serie.

    Equacao original (2a ordem):
        L*q'' + R*q' + q/C = V(t)

    Reducao para sistema de 1a ordem com y = [q, i]:
        y1' = y2           (dq/dt = i)
        y2' = (V(t) - R*y2 - y1/C) / L

    Parametros:
        t     : tempo (s)
        y     : vetor [q, i] — carga (C) e corrente (A)
        R     : resistencia (Ohm)
        L     : indutancia (H)  — deve ser > 0
        C     : capacitancia (F) — deve ser > 0
        fonte : funcao fonte(t) -> tensao V(t) em volts
    """
    if L <= 0:
        raise ValueError(f"Indutancia L deve ser positiva, recebeu L={L}")
    if C <= 0:
        raise ValueError(f"Capacitancia C deve ser positiva, recebeu C={C}")

    q, i = y
    dq_dt = i
    di_dt = (fonte(t) - R * i - q / C) / L
    return np.array([dq_dt, di_dt])
