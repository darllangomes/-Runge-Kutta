import numpy as np


def rk4(f, t0, tf, y0, dt, *args):
    """Integra dy/dt = f(t, y, *args) de t0 a tf com passo dt.

    Retorna (ts, ys): arrays com todos os pares (t, y(t)).
    Trata o caso em que (tf - t0) nao e multiplo de dt
    usando um passo final reduzido.
    """
    t = t0
    y = np.array(y0, dtype=float)
    ts = [t]
    ys = [y.copy()]

    while t < tf:
        h = min(dt, tf - t)

        k1 = f(t, y, *args)
        k2 = f(t + h / 2, y + k1 * h / 2, *args)
        k3 = f(t + h / 2, y + k2 * h / 2, *args)
        k4 = f(t + h, y + k3 * h, *args)

        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h

        ts.append(t)
        ys.append(y.copy())

    return np.array(ts), np.array(ys)
