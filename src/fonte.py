import numpy as np


def fonte_dc(V0):
    """Fonte contínua: V(t) = V0."""
    return lambda t: V0


def fonte_ac(amplitude, frequencia, offset=0.0):
    """Fonte alternada: V(t) = A*sin(2*pi*f*t) + offset."""
    return lambda t: amplitude * np.sin(2 * np.pi * frequencia * t) + offset
