"""
app_lista2.py — resolução passo a passo da Lista 2 da 3ª unidade (DCA-3701).

O que este arquivo faz
----------------------
Resolve, mostrando todas as contas intermediárias, os quatro exercícios da lista:
por meio deles você obtém a forma canônica controlável (itens 1a, 3a), a forma
canônica observável (item 4a) e o projeto do seguidor de referência para entrada
de degrau com pólos prescritos (itens 1b, 2, 3b e 4b).

A teoria e as fórmulas seguem o capítulo 7 do SisContr.pdf:
    formas canônicas .................. eqs. (7.11) e (7.12)
    controlabilidade ................... eq.  (7.9)
    fórmula de Ackermann ............... eq.  (7.15)
    seguidor de referência (degrau) .... eqs. (7.23) a (7.27), seção 7.10

O ponto central: no seguidor, quem é realimentado é o SISTEMA AUMENTADO
    A_a = [0  C; 0  A],   B_a = [0; B],
e não a planta original. São n+1 pólos a alocar (o integrador do modelo interno
consome um grau de liberdade). A lei de controle resultante é
    u(t) = k1 * ∫e(τ)dτ + k2 * x(t),   e(t) = y(t) - r(t).

Reaproveitamento do app_lista1.py
---------------------------------
As contas que não dependem do problema (formatação, matriz de controlabilidade,
polinômio desejado, q(A), Ackermann e autovalores) NÃO são redefinidas aqui:
são importadas de `app_lista1.py`. Este arquivo contém apenas o que é específico
da Lista 2 (formas canônicas a partir de G(s) e o seguidor de referência).

Como rodar
----------
    streamlit run app_lista2.py

Organização do arquivo (leia de cima para baixo)
------------------------------------------------
    1. IMPORTS                  — o que vem do app_lista1 e o que é próprio
    2. CONTAS DA LISTA 2       — sistema aumentado, malha fechada, formas canônicas
    3. BLOCOS DE APRESENTAÇÃO   — cada bloco desenha um passo na tela
    4. DADOS DA LISTA           — os polinômios e matrizes dos exercícios 1 a 4
    5. INTERFACE                — montagem das abas e chamada dos blocos

A última parte fica dentro de `if __name__ == "__main__":` para que o arquivo
possa ser importado em testes sem abrir a interface.
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
# O Streamlit executa o script com o diretório do arquivo como referência, mas
# garantimos isso explicitamente para que `import app_lista1` funcione também
# quando o app é iniciado a partir de outra pasta.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# `sympy` para conta exata e `streamlit` para a tela (mesmo motivo da Lista 1).
import sympy as sp
import streamlit as st

# Funções reaproveitadas da Lista 1. Todas são genéricas: recebem matrizes e
# pólos e não dependem dos dados daquela lista.
from app_lista1 import (
    s,                      # símbolo de Laplace
    latex_matriz,           # matriz sympy -> LaTeX
    mostrar_matriz,         # desenha "nome = matriz"
    passo,                  # cabeçalho "Passo n — título"
    matriz_controlabilidade,  # U = [B AB ... A^(n-1)B]
    polinomio_desejado,     # (s - p1)(s - p2)... expandido
    q_de_A,                 # q(A) = A^n + a1 A^(n-1) + ... + an I
    ackermann_k,            # K = -[0...1] U^-1 q_c(A)
    autovalores,            # autovalores ordenados, com multiplicidade
)


# ---------------------------------------------------------------------------
# 2. CONTAS DA LISTA 2 (o que é específico dela)
# ---------------------------------------------------------------------------
def sistema_aumentado_degrau(A, B, C):
    """Monta o sistema aumentado do seguidor de referência do tipo degrau.

    Da seção 7.10 do SisContr: definindo z = dx/dt, e = y - r e w = du/dt,
    como dr/dt = 0 tem-se e' = C*dx/dt. O resultado é

        d/dt [e; z] = [0  C; 0  A] [e; z] + [0; B] w

    que é uma dinâmica de n+1 estados. Retornamos (A_a, B_a).
    """
    n = A.shape[0]
    zero1 = sp.Matrix([[0]])
    Aa = sp.Matrix.vstack(
        sp.Matrix.hstack(zero1, C),                 # [0  C]  (1 x n+1)
        sp.Matrix.hstack(sp.zeros(n, 1), A),        # [0  A]  (n x n+1)
    )
    Ba = sp.Matrix.vstack(sp.Matrix([0]), B)        # [0; B]  (n+1 x 1)
    return Aa, Ba


def malha_fechada_seguidor(A, B, C, k1, k2):
    """Monta a matriz dinâmica da malha fechada do seguidor.

    Definindo x_{n+1} = ∫e dτ, tem-se x'_{n+1} = e = Cx - r, e o sistema
    fechado fica (SisContr, eq. 7.27):

        d/dt [x; x_{n+1}] = [A + B k2   B k1;  C   0] [x; x_{n+1}] + [0; -1] r

    Retornamos a matriz (n+1)x(n+1) cujos autovalores devem ser os pólos
    desejados.
    """
    inf = sp.Matrix([[0]])
    return sp.Matrix.vstack(
        sp.Matrix.hstack(A + B * k2, B * k1),
        sp.Matrix.hstack(C, inf),
    )


def alinhar_numerador(num, n):
    """Padroniza o numerador para a lista [β1, β2, ..., βn].

    num vem com os coeficientes do maior grau para o menor, mas pode ter
    grau menor que n-1 (ex.: G(s) = 3/(s^2+2s-4), num = [3], n = 2).
    Completamos com zeros à esquerda e devolvemos exatamente n posições.
    """
    num = list(num)
    if len(num) > n:
        raise ValueError("numerador de grau maior ou igual ao denominador?")
    return [sp.Rational(0)] * (n - len(num)) + num


def polinomio(coefs):
    """Converte [c0, c1, ..., ck] em c0*s^k + c1*s^(k-1) + ... + ck."""
    grau = len(coefs) - 1
    return sum(sp.sympify(c) * s ** (grau - i) for i, c in enumerate(coefs))


def mesmos_polos(autovals, polos, tol=1e-8):
    """Compara dois multiconjuntos de números complexos com tolerância.

    Usado na conferência final do seguidor: os autovalores de A_cl (calculados
    por sympy) devem coincidir com os pólos pedidos, a menos de arredondamento.
    Ordenamos por (parte real, parte imaginária) antes de comparar.
    """
    def chave(z):
        z = complex(sp.N(z))
        return (round(z.real, 9), round(z.imag, 9))
    a = sorted((complex(sp.N(v)) for v in autovals), key=chave)
    b = sorted((complex(sp.N(v)) for v in polos), key=chave)
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def canonica_controlavel(num, den):
    """Realização na forma canônica controlável a partir de G(s).

    den = [1, α1, ..., αn] (mônico), num = coeficientes do maior para o menor.
    Devolve (Ac, Bc, Cc) conforme a eq. (7.12): A tem 1's na superdiagonal e
    última linha = -coeficientes do denominador, Bc = e_n e Cc = [βn ... β1].
    """
    n = len(den) - 1
    beta = alinhar_numerador(num, n)

    Ac = sp.zeros(n, n)
    for i in range(n - 1):
        Ac[i, i + 1] = 1
    for j in range(n):
        Ac[n - 1, j] = -den[n - j]     # -αn ... -α1

    Bc = sp.Matrix([0] * (n - 1) + [1])
    Cc = sp.Matrix([list(reversed(beta))])   # [βn ... β1]
    return Ac, Bc, Cc


def canonica_observavel(num, den):
    """Realização na forma canônica observável a partir de G(s).

    É a transposta da controlável (eq. 7.11): Ao = Acᵀ, Bo = Ccᵀ, Co = Bcᵀ.
    """
    Ac, Bc, Cc = canonica_controlavel(num, den)
    return Ac.T, Cc.T, Bc.T


# ---------------------------------------------------------------------------
# 3. BLOCOS DE APRESENTAÇÃO
# Cada função abaixo desenha UM passo do raciocínio na tela.
# ---------------------------------------------------------------------------
def bloco_canonica(num, den, tipo):
    """Itens de forma canônica: lê α e β, monta (A, B, C) e confere G(s).

    `tipo` é "controlavel" ou "observavel".
    """
    n = len(den) - 1
    beta = alinhar_numerador(num, n)   # [β1 ... βn]

    passo(1, "Ler os coeficientes de G(s)")
    # Monta "α_1 = 3, α_2 = 7" em LaTeX de verdade (nada de Unicode no math mode).
    alfa_txt = ", \\; ".join(rf"\alpha_{{{i + 1}}} = {sp.latex(den[i + 1])}"
                             for i in range(n))
    beta_txt = ", \\; ".join(rf"\beta_{{{i + 1}}} = {sp.latex(beta[i])}"
                             for i in range(n))
    st.latex(alfa_txt + r" \qquad " + beta_txt)
    st.caption("α's vêm do denominador mônico (trocando o sinal ao entrar na última "
               "linha/coluna de A); β's vêm do numerador, do maior grau para o menor.")

    passo(2, "Montar A, B e C")
    if tipo == "controlavel":
        A, B, C = canonica_controlavel(num, den)
        st.caption("1's na superdiagonal, última linha = -coeficientes do denominador; "
                   "B = coluna unitária; C = [βn ... β1].")
    else:
        A, B, C = canonica_observavel(num, den)
        st.caption("Transposta da controlável: -coeficientes do denominador na última "
                   "coluna, B = [βn ... β1]ᵀ; C = [0 ... 0 1].")
    mostrar_matriz("A", A)
    mostrar_matriz("B", B)
    mostrar_matriz("C", C)

    passo(3, "Conferir que a realização reproduz G(s)")
    G = sp.simplify((C * (s * sp.eye(n) - A).inv() * B)[0, 0])
    st.latex(r"C(sI - A)^{-1}B = " + sp.latex(G))
    G_dada = sp.simplify(polinomio(num) / polinomio(den))
    if sp.simplify(G - G_dada) == 0:
        st.success("A realização reproduz a função de transferência dada.")
    else:
        st.error("A realização NÃO reproduz G(s) — confira os coeficientes.")
    return A, B, C


def bloco_seguidor(A, B, C, polos):
    """Projeto do seguidor de referência para entrada do tipo degrau.

    Roteiro (SisContr, seção 7.10): sistema aumentado → teste de
    controlabilidade → Ackermann em (A_a, B_a) → lei u = k1∫e + k2x →
    conferência dos autovalores da malha fechada.
    """
    n = A.shape[0]

    passo(1, "Montar o sistema aumentado (modelo interno do degrau)")
    st.caption("Como r(t) é degrau, ṙ = 0 e ė = Cẋ. Definindo z = ẋ e w = u̇, o erro e "
               "obedece à dinâmica aumentada abaixo, de dimensão n+1.")
    Aa, Ba = sistema_aumentado_degrau(A, B, C)
    mostrar_matriz("A_a", Aa)
    mostrar_matriz("B_a", Ba)

    passo(2, "Testar a controlabilidade do sistema aumentado")
    st.caption("O seguidor só existe se (A_a, B_a) for controlável — é o 'se for "
               "possível' do enunciado. Note que o teste é sobre o aumentado, "
               "com n+1 colunas.")
    Ua = matriz_controlabilidade(Aa, Ba)
    mostrar_matriz("U_a", Ua)
    det_Ua = sp.simplify(Ua.det())
    posto = Ua.rank()
    st.latex(r"\det(U_a) = " + sp.latex(det_Ua) +
             r"\qquad \operatorname{posto}(U_a) = " + str(posto))
    if posto < n + 1:
        st.error(f"posto(U_a) = {posto} < {n + 1} → sistema aumentado NÃO controlável. "
                 "Não existe seguidor de referência com os pólos pedidos.")
        return None
    st.success(f"posto(U_a) = {n + 1} → sistema aumentado controlável; "
               "os pólos podem ser alocados onde quisermos.")

    passo(3, "Polinômio característico desejado")
    pd = polinomio_desejado(polos)
    st.latex(r"\Delta_d(s) = " + sp.latex(pd))
    coefs = sp.Poly(pd, s).all_coeffs()
    st.latex(", \\quad ".join(rf"a_{{{i}}} = {sp.latex(coefs[i])}"
                              for i in range(1, len(coefs))))

    passo(4, "Cálculo de q_c(A_a)")
    st.caption("Mesma combinação do polinômio desejado, com a matriz A_a no lugar de s.")
    qc = q_de_A(Aa, coefs)
    st.latex(r"q_c(A_a) = " + latex_matriz(qc))

    passo(5, "Matriz U_a e sua inversa")
    mostrar_matriz("U_a^{-1}", Ua.inv())

    passo(6, "Ganho K do sistema aumentado")
    st.caption("K_a = -[0 ... 0 1]·U_a⁻¹·q_c(A_a), pela fórmula de Ackermann. "
               "O resultado se decompõe em k1 (escalar, do integrador) e k2 (vetor, da planta).")
    Ka = ackermann_k(Aa, Ba, polos)
    mostrar_matriz(r"K_a = \begin{bmatrix} k_1 & k_2 \end{bmatrix}", Ka)
    k1 = Ka[0]
    k2 = Ka[:, 1:]
    st.latex(r"k_1 = " + sp.latex(k1))
    mostrar_matriz("k_2", k2)

    passo(7, "Lei de controle (o que se implementa)")
    st.latex(r"u(t) = k_{1}\int_{0}^{t} e(\tau)\,d\tau + k_{2}\,x(t), \qquad e(t) = y(t) - r(t)")
    vetor_x = (r" \begin{bmatrix} " +
               r" \\ ".join(f"x_{{{i + 1}}}" for i in range(n)) +
               r" \end{bmatrix}")
    st.latex(r"u(t) = " + sp.latex(k1) + r" \int_{0}^{t} e(\tau)\,d\tau + " +
             sp.latex(k2) + vetor_x)

    passo(8, "Conferência: autovalores da malha fechada")
    st.caption("Montamos a matriz dinâmica do sistema fechado (eq. 7.27) e conferimos "
               "se os autovalores são os pólos pedidos.")
    M = malha_fechada_seguidor(A, B, C, k1, k2)
    mostrar_matriz("A_{cl}", M)
    eigs = autovalores(M)
    st.latex(r"\operatorname{eig}(A_{cl}) = " + sp.latex(eigs))
    if not mesmos_polos(eigs, polos):
        st.error("Os autovalores de A_cl NÃO coincidem com os pólos pedidos — "
                 "o ganho calculado está errado.")
    elif any(float(sp.re(sp.N(v))) >= 0 for v in eigs):
        st.warning("Pólos alocados corretamente, mas há pólo(s) com parte real ≥ 0: "
                   "a malha fechada não é estável e e(t) NÃO tende a zero.")
    else:
        st.success("Pólos de malha fechada do seguidor alocados corretamente. "
                   "O erro de rastreamento tende a zero (e(t) → 0).")
    return k1, k2


# ---------------------------------------------------------------------------
# 4. DADOS DA LISTA
# ---------------------------------------------------------------------------
# Exercício 1: G(s) = (3s² + 4s − 2) / (s³ + 3s² + 7s + 5).
# Coeficientes do maior grau para o menor; denominador mônico.
NUM1 = [3, 4, -2]
DEN1 = [1, 3, 7, 5]

# Exercício 2: sistema dado diretamente em espaço de estados.
A2 = sp.Matrix([[-6, 1, 0],
                [-11, 0, 1],
                [-6, 0, 0]])
B2 = sp.Matrix([2, 6, 2])
C2 = sp.Matrix([[1, 0, 0]])

# Exercício 3: G(s) = (2s + 1) / (s(s + 3)) = (2s + 1) / (s² + 3s + 0).
NUM3 = [2, 1]
DEN3 = [1, 3, 0]

# Exercício 4: G(s) = 3 / (s² + 2s − 4). Numerador constante (β1 = 0).
NUM4 = [3]
DEN4 = [1, 2, -4]

# Pólos pedidos nos itens. Note que −1 ± 0.1i é um par conjugado; escrevemos
# com fração exata I/10 para as contas saírem exatas.
POLOS_12 = [-1 + sp.I / 10, -1 - sp.I / 10, -5, -12]   # exercícios 1b e 2
POLOS_3 = [-1, -1, -1]                                 # exercício 3b
POLOS_4 = [-2, -2, -3]                                 # exercício 4b


# ---------------------------------------------------------------------------
# 5. INTERFACE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Lista 2 — 3ª unidade", layout="wide")

    st.title("Lista 2 da 3ª unidade — resolução passo a passo")
    st.caption("DCA-3701 · Projeto de Sistemas de Controle · fórmulas do capítulo 7 do SisContr.pdf")

    with st.sidebar:
        st.header("Convenções")
        st.markdown(
            "- Erro de rastreamento: $e(t) = y(t) - r(t)$\n"
            "- Lei de controle: $u(t) = k_1 \\int_0^t e(\\tau)\\,d\\tau + k_2\\,x(t)$\n"
            "- Sistema aumentado: $\\dot{a} = A_a a + B_a w$, com\n"
            "  $A_a = \\begin{bmatrix} 0 & C \\\\ 0 & A \\end{bmatrix}$, "
            "  $B_a = \\begin{bmatrix} 0 \\\\ B \\end{bmatrix}$\n"
            "- Malha fechada: $A_{cl} = \\begin{bmatrix} A + B k_2 & B k_1 \\\\ C & 0 \\end{bmatrix}$\n"
        )
        st.markdown("**Fórmula central (Ackermann no aumentado)**")
        st.latex(r"K_a = -[0\;\cdots\;1]\,U_a^{-1}q_c(A_a)")
        st.latex(r"U_a = [B_a\;\; A_a B_a\;\; \cdots \;\; A_a^{n}B_a]")
        st.divider()
        st.markdown("**O seguidor consome um grau de liberdade**")
        st.markdown(
            "São $n+1$ pólos a alocar (a planta tem $n$). Por isso os "
            "enunciados pedem 4 pólos para sistemas de 3ª ordem e 3 pólos "
            "para os de 2ª ordem."
        )
        st.divider()
        st.markdown("As funções de conta compartilhadas são importadas de "
                    "`app_lista1.py`; este app só implementa o que é "
                    "específico da Lista 2.")
        st.markdown("As contas matriciais são exatas (sympy); apenas os "
                    "autovalores de conferência são numéricos ou simbólicos.")

    aba1, aba2, aba3, aba4 = st.tabs(
        ["1 · G(s) cúbica", "2 · Sistema 3×3", "3 · G(s) 2ª ordem", "4 · G(s) observável"]
    )

    # ---------------- Exercício 1 ----------------
    with aba1:
        st.header("Exercício 1 — G(s) de 3ª ordem")
        st.latex(r"G(s) = \frac{3s^{2} + 4s - 2}{s^{3} + 3s^{2} + 7s + 5}")

        st.subheader("a) Forma canônica controlável")
        A1, B1, C1 = bloco_canonica(NUM1, DEN1, "controlavel")

        st.divider()

        st.subheader("b) Seguidor de referência para degrau — pólos "
                     "$-1{,}0 \\pm 0{,}1i$, $-5$ e $-12$")
        st.caption("Usamos a realização obtida no item (a). O teste do sistema "
                   "aumentado decide se a alocação é possível.")
        bloco_seguidor(A1, B1, C1, POLOS_12)

    # ---------------- Exercício 2 ----------------
    with aba2:
        st.header("Exercício 2 — sistema em espaço de estados")
        mostrar_matriz("A", A2)
        mostrar_matriz("B", B2)
        mostrar_matriz("C", C2)

        st.subheader("Seguidor de referência para degrau — pólos "
                     "$-1{,}0 \\pm 0{,}1i$, $-5$ e $-12$")
        st.caption("O sistema já vem em espaço de estados: não é preciso passar "
                   "pela forma canônica. O projeto é direto sobre (A, B, C).")
        bloco_seguidor(A2, B2, C2, POLOS_12)

    # ---------------- Exercício 3 ----------------
    with aba3:
        st.header("Exercício 3 — G(s) com pólo na origem")
        st.latex(r"G(s) = \frac{2s + 1}{s(s + 3)} = \frac{2s + 1}{s^{2} + 3s}")
        st.caption("Cuidado ao ler os coeficientes: o denominador é s² + 3s + **0**, "
                   "ou seja, α1 = 3 e α2 = 0.")

        st.subheader("a) Forma canônica controlável")
        A3, B3, C3 = bloco_canonica(NUM3, DEN3, "controlavel")

        st.divider()

        st.subheader("b) Seguidor de referência para degrau — todos os pólos em $-1$")
        bloco_seguidor(A3, B3, C3, POLOS_3)

    # ---------------- Exercício 4 ----------------
    with aba4:
        st.header("Exercício 4 — G(s) com numerador constante")
        st.latex(r"G(s) = \frac{3}{s^{2} + 2s - 4}")

        st.subheader("a) Forma canônica observável")
        st.caption("Numerador constante: β1 = 0 (não há termo em s) e β2 = 3.")
        A4, B4, C4 = bloco_canonica(NUM4, DEN4, "observavel")

        st.divider()

        st.subheader("b) Seguidor de referência para degrau — pólos $-2$, $-2$ e $-3$")
        st.caption("Usamos a realização observável obtida no item (a). "
                   "Como G(s) não tem cancelamento pólo-zero, ela é mínima, e a "
                   "propriedade de ser controlável independe da realização.")
        bloco_seguidor(A4, B4, C4, POLOS_4)
