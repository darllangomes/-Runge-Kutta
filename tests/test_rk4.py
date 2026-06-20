"""Testes de validacao do simulador RLC com RK4."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rk4 import rk4
from modelo import f
from fonte import fonte_dc, fonte_ac


# ============================================================
# Fase 1 — Checkpoint: f(0, [0,0], R=1, L=1, C=1, V=1) = [0, 1]
# ============================================================
class TestModelo:
    def test_checkpoint_fase1(self):
        resultado = f(0, np.array([0.0, 0.0]), R=1, L=1, C=1, fonte=lambda t: 1.0)
        np.testing.assert_array_almost_equal(resultado, [0.0, 1.0])

    def test_L_invalido(self):
        with pytest.raises(ValueError):
            f(0, np.array([0.0, 0.0]), R=1, L=0, C=1, fonte=lambda t: 1.0)

    def test_C_invalido(self):
        with pytest.raises(ValueError):
            f(0, np.array([0.0, 0.0]), R=1, L=1, C=-1, fonte=lambda t: 1.0)


# ============================================================
# Fase 2 — Validacao do RK4
# ============================================================
class TestRK4:
    def test_decaimento_exponencial(self):
        """dy/dt = -y, y(0)=1 => y(t) = e^(-t). Erro < 1e-6 em t=5."""
        ts, ys = rk4(lambda t, y: -y, 0, 5, [1.0], 0.1)
        erro = abs(ys[-1, 0] - np.exp(-5))
        assert erro < 1e-6, f"Erro muito alto: {erro}"

    def test_convergencia_4a_ordem(self):
        """Reduzir dt pela metade deve reduzir o erro ~16x."""
        def resolver(dt):
            ts, ys = rk4(lambda t, y: -y, 0, 1, [1.0], dt)
            return abs(ys[-1, 0] - np.exp(-1))

        erro1 = resolver(0.1)
        erro2 = resolver(0.05)
        razao = erro1 / erro2
        assert razao > 12, f"Razao de convergencia {razao:.1f} < 12 (esperado ~16)"

    def test_intervalo_nao_multiplo(self):
        """t0=0, tf=1, dt=0.3 => passos em [0, 0.3, 0.6, 0.9, 1.0]."""
        ts, ys = rk4(lambda t, y: -y, 0, 1, [1.0], 0.3)
        esperado = [0.0, 0.3, 0.6, 0.9, 1.0]
        np.testing.assert_array_almost_equal(ts, esperado)

    def test_dt_maior_que_intervalo(self):
        """dt > (tf - t0) => um unico passo."""
        ts, ys = rk4(lambda t, y: -y, 0, 0.5, [1.0], 2.0)
        assert len(ts) == 2
        assert ts[-1] == pytest.approx(0.5)


# ============================================================
# Fase 3 — Fontes
# ============================================================
class TestFontes:
    def test_dc(self):
        assert fonte_dc(5.0)(123.4) == 5.0

    def test_ac_em_zero(self):
        assert fonte_ac(10, 60, 2)(0) == pytest.approx(2.0)

    def test_ac_pico(self):
        """No pico (t = 1/(4f)), V = A + offset."""
        A, freq, off = 10, 60, 2
        t_pico = 1 / (4 * freq)
        assert fonte_ac(A, freq, off)(t_pico) == pytest.approx(A + off)


# ============================================================
# Fase 6 — Validacao com SciPy e casos extremos
# ============================================================
class TestValidacaoSciPy:
    def test_comparacao_scipy(self):
        """Diferenca entre RK4 manual e SciPy deve ser desprezivel."""
        from scipy.integrate import solve_ivp

        R, L, C = 50.0, 1e-3, 1e-6
        V = fonte_dc(10.0)
        t0, tf = 0.0, 5e-3
        dt = 1e-6
        y0 = [0.0, 0.0]

        ts, ys = rk4(f, t0, tf, y0, dt, R, L, C, V)

        sol = solve_ivp(
            lambda t, y: f(t, y, R, L, C, V),
            [t0, tf], y0, method="RK45",
            rtol=1e-9, atol=1e-12, dense_output=True,
        )

        i_scipy = sol.sol(ts)[1]
        i_rk4 = ys[:, 1]
        erro_max = np.max(np.abs(i_rk4 - i_scipy))
        assert erro_max < 1e-3, f"Erro maximo vs SciPy: {erro_max}"


class TestCasosExtremos:
    def test_R_zero_oscilacao_pura(self):
        """R=0 (LC puro): amplitude deve se manter aproximadamente constante."""
        R, L, C = 0.0, 1e-3, 1e-6
        V = fonte_dc(10.0)
        T0 = 2 * np.pi * np.sqrt(L * C)
        dt = T0 / 200
        tf = T0 * 20

        ts, ys = rk4(f, 0, tf, [0.0, 0.0], dt, R, L, C, V)
        corrente = ys[:, 1]

        # Apos o transitorio (primeiros 2 ciclos), a amplitude deve ser estavel
        inicio = int(len(corrente) * 0.3)
        picos = np.abs(corrente[inicio:])
        amp_max = np.max(picos)
        amp_min_picos = np.max(np.abs(corrente[int(len(corrente) * 0.8):]))
        razao = amp_min_picos / amp_max if amp_max > 0 else 1.0
        assert razao > 0.95, f"Amplitude caiu muito: razao = {razao:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
