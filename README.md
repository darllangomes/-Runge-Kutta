# Simulador de Circuito RLC Série — Runge-Kutta 4ª Ordem

Simulador interativo de um circuito RLC série, resolvido numericamente pelo método de Runge-Kutta de 4ª ordem (RK4). Desenvolvido para a 2ª VA de Física Aplicada à Computação (UFRPE, 2026).

## Funcionalidades

- Integração numérica via RK4 com passo final reduzido automaticamente
- Redução da EDO de 2ª ordem para sistema de 1ª ordem
- Fonte de tensão DC (constante) ou AC (senoidal com amplitude, frequência e offset)
- Interface gráfica interativa com sliders para ajustar R, L, C e parâmetros da fonte
- Gráfico de corrente × tempo atualizado em tempo real
- Visualização dos 3 regimes de amortecimento e ressonância AC

## Requisitos

- Python 3.8+
- numpy
- matplotlib
- scipy (apenas para validação nos testes)

## Instalação

```bash
git clone https://github.com/darllangomes/-Runge-Kutta.git
cd -Runge-Kutta
pip install -r requirements.txt
```

## Como usar

```bash
python src/main.py
```

A janela interativa abre com:

| Controle | Função |
|----------|--------|
| Slider R | Resistência (0–1000 Ω) |
| Slider L | Indutância (0,1–10 mH) |
| Slider C | Capacitância (0,1–10 µF) |
| Radio DC/AC | Tipo de fonte |
| Slider V0 | Tensão DC (V) |
| Sliders Ampl/Freq/Offset | Parâmetros da fonte AC |

### Dicas para demonstração

- **Subamortecido:** R baixo (ex: 10 Ω) → corrente oscila com decaimento
- **Crítico:** R ≈ 2√(L/C) ≈ 63 Ω (com L=1 mH, C=1 µF) → pico único, retorno rápido
- **Superamortecido:** R alto (ex: 300 Ω) → decaimento lento, sem oscilação
- **Ressonância AC:** ajustar frequência para f₀ = 1/(2π√(LC)) → amplitude máxima

## Testes

```bash
pytest tests/test_rk4.py -v
```

Cobertura: modelo físico, convergência 4ª ordem do RK4, fontes DC/AC, comparação com SciPy, casos extremos (R=0, intervalos não múltiplos, parâmetros inválidos).

## Estrutura do projeto

```
src/
├── main.py      # Interface interativa (matplotlib widgets)
├── modelo.py    # f(t, y): sistema de 1ª ordem do circuito RLC
├── rk4.py       # Integrador Runge-Kutta de 4ª ordem
└── fonte.py     # Geradores de fonte DC e AC
tests/
└── test_rk4.py  # Suite de validação com pytest
```

## Fundamentação

A equação do circuito RLC série pela Lei de Kirchhoff:

```
L·q̈ + R·q̇ + q/C = V(t)
```

É reduzida ao sistema de 1ª ordem com y = [q, i]:

```
ẏ₁ = y₂
ẏ₂ = (V(t) − R·y₂ − y₁/C) / L
```

O RK4 integra este sistema com erro global O(Δt⁴).

## Licença

MIT
