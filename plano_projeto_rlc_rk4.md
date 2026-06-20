# Plano de Implementação — Simulador de Circuito RLC com Runge–Kutta de 4ª Ordem

**Disciplina:** Física Aplicada à Computação — 2ª VA (UFRPE, 2026)
**Linguagem:** Python 3
**Entregável:** Simulador com gráfico de corrente × tempo, parâmetros R, L, C ajustáveis e fonte DC/AC configurável

---

## Visão geral

| Fase | Objetivo | Resultado verificável |
|------|----------|----------------------|
| 0 | Ambiente e estrutura | Projeto roda "hello world" com dependências instaladas |
| 1 | Modelagem física | Função `f(t, y)` testada manualmente |
| 2 | Núcleo RK4 | Integrador validado contra solução analítica |
| 3 | Fontes DC e AC | V(t) configurável nos dois modos |
| 4 | Simulação completa + gráfico | Gráfico corrente × tempo correto nos 3 regimes |
| 5 | Interface interativa | Todos os requisitos do enunciado atendidos |
| 6 | Validação e robustez | Casos extremos tratados, comparação com SciPy |
| 7 | Documentação e entrega | README, relatório, código comentado |

Cada fase termina com um **checkpoint** — um critério objetivo de "pronto". Conclua uma fase antes de avançar.

---

## Fase 0 — Ambiente e estrutura do projeto

**Objetivo:** preparar o terreno para não perder tempo depois.

### Tarefas

- [ ] Criar repositório/pasta do projeto (sugestão: versionar com Git desde o início)
- [ ] Criar ambiente virtual: `python -m venv .venv`
- [ ] Instalar dependências: `pip install numpy matplotlib`
  - SciPy é opcional (usado só na Fase 6 para validação): `pip install scipy`
- [ ] Criar a estrutura de arquivos:

```
projeto-rlc/
├── src/
│   ├── fonte.py        # Funções da fonte de tensão (DC e AC)
│   ├── modelo.py       # Equação do circuito: f(t, y)
│   ├── rk4.py          # Integrador Runge–Kutta de 4ª ordem
│   └── main.py         # Interface + gráfico
├── tests/
│   └── test_rk4.py     # Validações da Fase 6
├── requirements.txt
└── README.md
```

> **Alternativa válida:** um único arquivo `simulador_rlc.py` com seções bem separadas. Para projeto de disciplina, ambos funcionam — a estrutura modular facilita testar cada parte isoladamente, que é exatamente o espírito deste plano faseado.

### Checkpoint ✅
`python src/main.py` executa sem erro (mesmo que só imprima algo).

---

## Fase 1 — Modelagem física: a função f(t, y)

**Objetivo:** transformar a equação de 2ª ordem do circuito em um sistema de 1ª ordem (o "ajuste" exigido pelo enunciado) e implementá-lo.

### Teoria

A Lei de Kirchhoff das tensões no circuito RLC série dá:

```
L·q̈ + R·q̇ + q/C = V(t)
```

O RK4 só resolve `ẏ = f(t, y)`. Definindo o vetor de estado:

```
y = [y₁, y₂] = [q, i]      (carga e corrente)
```

O sistema de 1ª ordem equivalente é:

```
ẏ₁ = y₂
ẏ₂ = (V(t) − R·y₂ − y₁/C) / L
```

O lado direito depende só de `t` e `y` — formato exato exigido pelo enunciado (sem derivadas no lado direito).

### Tarefas

- [ ] Implementar em `modelo.py`:

```python
import numpy as np

def f(t, y, R, L, C, fonte):
    """Lado direito do sistema de 1ª ordem do circuito RLC série.

    y[0] = q (carga no capacitor, em C)
    y[1] = i (corrente no circuito, em A)
    fonte: função fonte(t) -> tensão V(t)
    """
    q, i = y
    dq_dt = i
    di_dt = (fonte(t) - R * i - q / C) / L
    return np.array([dq_dt, di_dt])
```

- [ ] Teste de mesa manual: com `R=1, L=1, C=1, V=1, q=0, i=0`, conferir que `f` retorna `[0, 1]`
- [ ] Conferir unidades no docstring (R em Ω, L em H, C em F, V em volts)

### Checkpoint ✅
Chamar `f(0, np.array([0.0, 0.0]), R=1, L=1, C=1, fonte=lambda t: 1.0)` retorna `[0.0, 1.0]`.

---

## Fase 2 — Núcleo do RK4

**Objetivo:** implementar o integrador exatamente como no pseudocódigo do enunciado, mas vetorizado (funciona para sistemas), com os dois detalhes obrigatórios:
1. tratar `tF − tI` não múltiplo de `Δt` (último passo reduzido);
2. salvar **todos** os pares `(t, y(t))`.

### Teoria

Para avançar de `t` a `t + Δt`, o RK4 amostra a inclinação em 4 pontos:

| Estágio | Onde | Usando |
|---------|------|--------|
| k₁ | início do passo | estado atual |
| k₂ | meio do passo | k₁ |
| k₃ | meio do passo (refinado) | k₂ |
| k₄ | fim do passo | k₃ |

Atualização: `y ← y + (Δt/6)(k₁ + 2k₂ + 2k₃ + k₄)`.
Erro global O(Δt⁴): reduzir Δt pela metade reduz o erro ~16×.

Como `y` é um `np.array`, o mesmo código serve para o sistema de 2 equações — as operações são elemento a elemento.

### Tarefas

- [ ] Implementar em `rk4.py`:

```python
import numpy as np

def rk4(f, t0, tf, y0, dt, *args):
    """Integra ẏ = f(t, y, *args) de t0 a tf com passo dt.

    Retorna (ts, ys): arrays com todos os pares (t, y(t)).
    Trata o caso em que (tf - t0) não é múltiplo de dt
    usando um passo final reduzido.
    """
    t = t0
    y = np.array(y0, dtype=float)
    ts = [t]
    ys = [y.copy()]

    while t < tf:
        # Detalhe obrigatório 1: último passo reduzido
        h = min(dt, tf - t)

        k1 = f(t, y, *args)
        k2 = f(t + h / 2, y + k1 * h / 2, *args)
        k3 = f(t + h / 2, y + k2 * h / 2, *args)
        k4 = f(t + h, y + k3 * h, *args)

        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h

        # Detalhe obrigatório 2: salvar cada par (t, y)
        ts.append(t)
        ys.append(y.copy())

    return np.array(ts), np.array(ys)
```

- [ ] **Validação analítica obrigatória** antes de prosseguir — resolver `ẏ = -y, y(0)=1`, cuja solução exata é `e^(−t)`:

```python
ts, ys = rk4(lambda t, y: -y, 0, 5, [1.0], 0.1)
erro = abs(ys[-1, 0] - np.exp(-5))
print(erro)  # deve ser < 1e-6
```

- [ ] Testar convergência de 4ª ordem: rodar com `dt` e `dt/2`; o erro deve cair ~16×
- [ ] Testar intervalo não múltiplo: `t0=0, tf=1, dt=0.3` → último t deve ser exatamente `1.0` e os passos `[0, 0.3, 0.6, 0.9, 1.0]`

### Checkpoint ✅
Os três testes acima passam. **Não avance com o RK4 errado** — todo o resto depende dele.

---

## Fase 3 — Fontes de tensão (DC e AC)

**Objetivo:** implementar V(t) configurável nos dois modos exigidos.

### Requisitos do enunciado cobertos
- Escolher entre fonte contínua e alternada ✔
- DC: tensão ajustável ✔
- AC: amplitude, frequência e offset ajustáveis ✔

### Tarefas

- [ ] Implementar em `fonte.py`:

```python
import numpy as np

def fonte_dc(V0):
    """Fonte contínua: V(t) = V0."""
    return lambda t: V0

def fonte_ac(amplitude, frequencia, offset=0.0):
    """Fonte alternada: V(t) = A·sin(2πft) + offset."""
    return lambda t: amplitude * np.sin(2 * np.pi * frequencia * t) + offset
```

- [ ] Teste rápido: `fonte_ac(10, 60, 2)(0)` → `2.0`; `fonte_dc(5)(123.4)` → `5.0`

> **Design:** retornar *funções* (closures) mantém `f(t, y)` desacoplada do tipo de fonte — o modelo só chama `fonte(t)` sem saber se é DC ou AC. Isso simplifica a Fase 5.

### Checkpoint ✅
Ambas as fontes retornam os valores esperados para t arbitrário.

---

## Fase 4 — Simulação completa + gráfico corrente × tempo

**Objetivo:** juntar as peças e gerar o gráfico exigido pelo enunciado, conferindo a física nos três regimes.

### Tarefas

- [ ] Montar a simulação em `main.py` (versão sem interface ainda):

```python
import numpy as np
import matplotlib.pyplot as plt
from modelo import f
from rk4 import rk4
from fonte import fonte_dc, fonte_ac

# Parâmetros de exemplo
R, L, C = 50.0, 1e-3, 1e-6        # Ω, H, F
V = fonte_dc(10.0)
t0, tf = 0.0, 5e-3                 # 5 ms de simulação
dt = 1e-6

y0 = [0.0, 0.0]                    # q(0) = 0, i(0) = 0 (circuito desenergizado)
ts, ys = rk4(f, t0, tf, y0, dt, R, L, C, V)

plt.plot(ts * 1e3, ys[:, 1] * 1e3)  # eixos em ms e mA
plt.xlabel("Tempo (ms)")
plt.ylabel("Corrente (mA)")
plt.title("Circuito RLC série — corrente × tempo")
plt.grid(True)
plt.show()
```

- [ ] **Verificação física dos 3 regimes** (fonte DC, L = 1 mH, C = 1 µF → R crítico = 2√(L/C) ≈ 63,2 Ω):
  - [ ] **Subamortecido** (R = 10 Ω): corrente oscila com amplitude decaindo
  - [ ] **Crítico** (R ≈ 63,2 Ω): pico único, decai sem oscilar, retorno mais rápido
  - [ ] **Superamortecido** (R = 300 Ω): pico único, decaimento lento, sem oscilação
- [ ] **Verificação com AC:** frequência natural f₀ = 1/(2π√(LC)) ≈ 5,03 kHz para os valores acima
  - [ ] Fonte em f = f₀ → amplitude de corrente máxima em regime permanente (ressonância)
  - [ ] Fonte em f ≪ f₀ ou f ≫ f₀ → amplitude bem menor
- [ ] **Sanidade DC:** em regime permanente com fonte DC, a corrente deve tender a **zero** (capacitor carregado bloqueia DC)

### Checkpoint ✅
Os três regimes e a ressonância aparecem corretamente nos gráficos. Salve essas figuras — servem para o relatório.

---

## Fase 5 — Interface interativa

**Objetivo:** atender aos requisitos "deve ser possível escolher/ajustar" de forma demonstrável.

### Decisão de escopo (escolha UMA)

| Opção | Esforço | Resultado |
|-------|---------|-----------|
| A. CLI com `input()`/`argparse` | Baixo | Cumpre requisitos, demo menos fluida |
| B. Matplotlib widgets (Sliders + RadioButtons) | Médio | Gráfico atualiza ao vivo, zero dependência extra |
| C. Streamlit | Médio | Interface web, ótima para apresentação |

**Recomendação: Opção B** — sem dependências além do que já está instalado, e o professor vê os parâmetros sendo ajustados em tempo real.

### Tarefas (Opção B)

- [ ] Layout: gráfico no topo; abaixo, sliders e botões de rádio
- [ ] Controles:
  - [ ] Sliders: R, L, C (escala log ou faixas adequadas: R em 1–1000 Ω, L em 0,1–10 mH, C em 0,1–10 µF)
  - [ ] RadioButtons: modo DC / AC
  - [ ] Modo DC: slider de tensão V₀
  - [ ] Modo AC: sliders de amplitude, frequência e offset
  - [ ] (Opcional) Sliders de tf e Δt
- [ ] Callback: qualquer mudança → reexecuta `rk4` → atualiza a curva (`line.set_data` + `ax.relim()` + `ax.autoscale_view()` + `fig.canvas.draw_idle()`)
- [ ] Mostrar/ocultar sliders conforme o modo selecionado (ou manter todos visíveis e ignorar os irrelevantes — mais simples)
- [ ] **Δt automático sugerido:** calcular `dt = T0/100` com `T0 = 2π√(LC)` para evitar que o usuário escolha um passo que faça a simulação divergir ou ficar imprecisa

### Esqueleto da interface

```python
from matplotlib.widgets import Slider, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.45)  # espaço para os controles

line, = ax.plot([], [])

ax_R = plt.axes([0.15, 0.30, 0.7, 0.03])
s_R = Slider(ax_R, "R (Ω)", 1, 1000, valinit=50)
# ... demais sliders e RadioButtons

def atualizar(_):
    # ler valores dos widgets, montar fonte, rodar rk4, atualizar line
    ...

s_R.on_changed(atualizar)
```

### Checkpoint ✅
Percorrer a lista de requisitos do enunciado, um a um, operando apenas a interface:
- [ ] Gráfico corrente × tempo ✔
- [ ] Escolher R, C e L ✔
- [ ] Escolher DC ou AC ✔
- [ ] DC: ajustar tensão ✔
- [ ] AC: ajustar amplitude, frequência e offset ✔

---

## Fase 6 — Validação e robustez

**Objetivo:** garantir que o simulador não quebra e que o resultado é numericamente confiável.

### Tarefas

- [ ] **Comparação com SciPy** (só validação, não entra no entregável principal):

```python
from scipy.integrate import solve_ivp
sol = solve_ivp(lambda t, y: f(t, y, R, L, C, V), [t0, tf], y0,
                method="RK45", rtol=1e-9, atol=1e-12, dense_output=True)
# comparar sol.sol(ts)[1] com ys[:, 1] — diferença deve ser desprezível
```

- [ ] **Casos extremos:**
  - [ ] `tf − t0` não múltiplo de `dt` (ex.: tf = 1 ms, dt = 0,3 ms) → termina exatamente em tf
  - [ ] `dt > tf − t0` → executa um único passo de tamanho `tf − t0`
  - [ ] R = 0 (oscilação sem amortecimento — LC puro): amplitude deve se manter constante
  - [ ] Valores inválidos: L ≤ 0 ou C ≤ 0 → mensagem de erro clara (`raise ValueError`), não crash
- [ ] **Teste de convergência documentado:** tabela erro × Δt mostrando a queda de 4ª ordem (bom material para o relatório)
- [ ] (Se modularizou) Transformar as validações em testes no `tests/test_rk4.py` rodáveis com `pytest`

### Checkpoint ✅
Todos os casos extremos passam; diferença vs. SciPy < 1e-6 nos casos de teste.

---

## Fase 7 — Documentação e entrega

**Objetivo:** empacotar o trabalho de forma profissional.

### Tarefas

- [ ] **README.md** com: descrição, como instalar (`pip install -r requirements.txt`), como executar, prints da interface
- [ ] **Código comentado**, com destaque (docstrings) para os dois pontos que o enunciado enfatiza:
  - a redução de ordem (2ª → sistema de 1ª)
  - o tratamento do último passo quando o intervalo não é múltiplo de Δt
- [ ] **Relatório/apresentação** (se exigido) cobrindo:
  - [ ] Dedução da equação do circuito via Lei de Kirchhoff
  - [ ] Analogia mecânica (massa–mola–amortecedor) — enriquece a discussão
  - [ ] Redução de ordem (o "ajuste" do enunciado)
  - [ ] Explicação do RK4 e por que é de 4ª ordem (com a tabela de convergência da Fase 6)
  - [ ] Gráficos dos 3 regimes de amortecimento + ressonância (figuras da Fase 4)
- [ ] Revisão final: rodar do zero em ambiente limpo (`python -m venv` novo) para garantir que o `requirements.txt` está completo

### Checkpoint ✅
Outra pessoa consegue clonar, instalar e rodar o projeto seguindo apenas o README.

---

## Referência rápida — fórmulas do projeto

| Grandeza | Fórmula |
|----------|---------|
| Equação do circuito | L·q̈ + R·q̇ + q/C = V(t) |
| Sistema de 1ª ordem | ẏ₁ = y₂ ; ẏ₂ = (V(t) − R·y₂ − y₁/C)/L |
| Resistência crítica | R_c = 2√(L/C) |
| Frequência natural | f₀ = 1/(2π√(LC)) |
| Período natural | T₀ = 2π√(LC) |
| Passo recomendado | Δt ≈ T₀/100 |
| Fonte DC | V(t) = V₀ |
| Fonte AC | V(t) = A·sin(2πft) + offset |

## Estimativa de esforço

| Fase | Tempo estimado |
|------|----------------|
| 0 | 30 min |
| 1 | 30 min |
| 2 | 1h30 (validação inclusa) |
| 3 | 20 min |
| 4 | 1h30 (verificações físicas) |
| 5 | 2–3h (opção B) |
| 6 | 1h30 |
| 7 | 1–2h |
| **Total** | **~9–11h** |
