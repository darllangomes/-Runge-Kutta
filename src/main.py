import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

from modelo import f
from rk4 import rk4
from fonte import fonte_dc, fonte_ac


def simular(R, L, C, fonte, tf, dt):
    """Executa a simulacao RLC e retorna (ts_ms, corrente_mA)."""
    y0 = [0.0, 0.0]
    ts, ys = rk4(f, 0.0, tf, y0, dt, R, L, C, fonte)
    return ts * 1e3, ys[:, 1] * 1e3  # ms e mA


def main():
    # --- Valores iniciais ---
    R0, L0_mH, C0_uF = 50.0, 1.0, 1.0
    V0_dc = 10.0
    A0_ac, freq0_ac, off0_ac = 10.0, 5000.0, 0.0
    modo_inicial = "DC"

    # --- Figura e eixo principal ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.10, bottom=0.52, right=0.88, top=0.93)
    ax.set_xlabel("Tempo (ms)")
    ax.set_ylabel("Corrente (mA)")
    ax.set_title("Circuito RLC serie — corrente x tempo")
    ax.grid(True)
    line, = ax.plot([], [], lw=1.5, color="tab:blue")

    # --- Helpers ---
    def calc_dt(L, C):
        T0 = 2 * np.pi * np.sqrt(L * C)
        return T0 / 100

    def calc_tf(L, C):
        T0 = 2 * np.pi * np.sqrt(L * C)
        return max(T0 * 8, 1e-4)

    # --- Funcao de atualizacao ---
    def atualizar(_=None):
        R = s_R.val
        L = s_L.val * 1e-3     # mH -> H
        C = s_C.val * 1e-6     # uF -> F
        dt = calc_dt(L, C)
        tf = calc_tf(L, C)

        modo = radio.value_selected
        if modo == "DC":
            fonte = fonte_dc(s_V0.val)
        else:
            fonte = fonte_ac(s_A.val, s_freq.val, s_off.val)

        ts_ms, i_mA = simular(R, L, C, fonte, tf, dt)
        line.set_data(ts_ms, i_mA)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    # --- Sliders ---
    cor_slider = "lightgoldenrodyellow"

    ax_R = plt.axes([0.15, 0.40, 0.55, 0.025], facecolor=cor_slider)
    s_R = Slider(ax_R, "R (Ohm)", 0.0, 1000.0, valinit=R0, valstep=1.0)

    ax_L = plt.axes([0.15, 0.36, 0.55, 0.025], facecolor=cor_slider)
    s_L = Slider(ax_L, "L (mH)", 0.1, 10.0, valinit=L0_mH, valstep=0.1)

    ax_C = plt.axes([0.15, 0.32, 0.55, 0.025], facecolor=cor_slider)
    s_C = Slider(ax_C, "C (uF)", 0.1, 10.0, valinit=C0_uF, valstep=0.1)

    # DC
    ax_V0 = plt.axes([0.15, 0.24, 0.55, 0.025], facecolor=cor_slider)
    s_V0 = Slider(ax_V0, "V0 (V)", 0.0, 50.0, valinit=V0_dc, valstep=0.5)

    # AC
    ax_A = plt.axes([0.15, 0.16, 0.55, 0.025], facecolor=cor_slider)
    s_A = Slider(ax_A, "Ampl (V)", 0.0, 50.0, valinit=A0_ac, valstep=0.5)

    ax_freq = plt.axes([0.15, 0.12, 0.55, 0.025], facecolor=cor_slider)
    s_freq = Slider(ax_freq, "Freq (Hz)", 100, 20000, valinit=freq0_ac, valstep=100)

    ax_off = plt.axes([0.15, 0.08, 0.55, 0.025], facecolor=cor_slider)
    s_off = Slider(ax_off, "Offset (V)", -20.0, 20.0, valinit=off0_ac, valstep=0.5)

    # --- Radio button DC/AC ---
    ax_radio = plt.axes([0.78, 0.08, 0.15, 0.12], facecolor=cor_slider)
    radio = RadioButtons(ax_radio, ("DC", "AC"), active=0)

    # --- Conectar callbacks ---
    for s in [s_R, s_L, s_C, s_V0, s_A, s_freq, s_off]:
        s.on_changed(atualizar)
    radio.on_clicked(atualizar)

    # --- Plot inicial ---
    atualizar()
    plt.show()


if __name__ == "__main__":
    main()
