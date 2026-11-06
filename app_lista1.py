"""
app_lista1.py — resolução passo a passo da Lista 1 da 3ª unidade (DCA-3701).

O que este arquivo faz
----------------------
Resolve, mostrando todas as contas intermediárias, os quatro exercícios da lista:
polinômio característico e estabilidade, matriz de controlabilidade, matriz de
observabilidade, projeto de realimentação de estados (ganho K) e projeto de
observador de estados (ganho L).

A teoria e as fórmulas seguem o capítulo 7 do SisContr.pdf:
    K = -[0 ... 1] U^-1 q_c(A),   U = [B AB ... A^(n-1)B]
    L = q_L(A) V^-1 [0 ... 1]^T,  V = [C; CA; ...; CA^(n-1)]

Como rodar
----------
    streamlit run app_lista1.py

Organização do arquivo (leia de cima para baixo)
------------------------------------------------
    1. IMPORTS E SÍMBOLO s        — o que é importado e por quê
    2. FORMATAÇÃO                 — funções que só transformam número em LaTeX
    3. CONTAS DE CONTROLE         — as funções que fazem a matemática de fato
    4. BLOCOS DE APRESENTAÇÃO     — cada bloco desenha um passo na tela
    5. DADOS DA LISTA             — as matrizes dos exercícios 1 a 4
    6. INTERFACE                  — montagem das abas e chamada dos blocos

A última parte fica dentro de `if __name__ == "__main__":` para que o arquivo
possa ser importado em testes sem abrir a interface.
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS E SÍMBOLO
# ---------------------------------------------------------------------------
# `sympy` é a biblioteca de matemática simbólica. Usamos ela (e não numpy)
# porque aqui interessa o resultado EXATO: 8/3 em vez de 2.6666667, e o
# determinante -81 em vez de -81.00000000000003.
import sympy as sp

# `streamlit` é a biblioteca que transforma este script em uma página web.
# Cada chamada st.algo(...) desenha um elemento na página, na ordem em que
# aparece no código.
import streamlit as st

# Símbolo da variável de Laplace. `sp.symbols("s")` cria um objeto simbólico
# chamado s, que se comporta como uma variável algébrica: pode ser somado,
# multiplicado, derivado etc., sem valor numérico definido.
s = sp.symbols("s")


# ---------------------------------------------------------------------------
# 2. FORMATAÇÃO (nada de matemática aqui, só conversão para LaTeX)
# ---------------------------------------------------------------------------
def latex_matriz(M):
    """Converte uma matriz sympy em texto LaTeX no formato [ ... ].

    `sp.latex` é o tradutor do sympy: recebe uma expressão e devolve a string
    LaTeX correspondente. Os argumentos mat_str e mat_delim escolhem o estilo
    da matriz: mat_str="bmatrix" usa colchetes, mat_delim="" remove os
    delimitadores extras que o sympy colocaria em volta.
    """
    return sp.latex(M, mat_str="bmatrix", mat_delim="")


def mostrar_matriz(nome, M):
    """Desenha uma matriz na tela precedida do nome dela.

    st.latex(...) renderiza uma string LaTeX como fórmula na página.
    A string é montada por concatenação (nome + " = " + corpo) porque dentro de
    f-string não podemos usar barras invertidas com segurança em todas as
    versões do Python.
    """
    st.latex(nome + " = " + latex_matriz(M))


# ---------------------------------------------------------------------------
# 3. CONTAS DE CONTROLE (o núcleo matemático do programa)
# ---------------------------------------------------------------------------
def polinomio_caracteristico(A):
    """p(s) = det(sI - A), devolvido como expressão sympy já expandida.

    `s * sp.eye(n)` monta a matriz sI: a identidade n x n multiplicada pelo
    símbolo s. Subtrair A dá sI - A, e `.det()` calcula o determinante
    simbólico. `sp.expand` só arruma o resultado na forma de polinômio.
    """
    n = A.shape[0]                      # A.shape devolve (linhas, colunas)
    return sp.expand((s * sp.eye(n) - A).det())


def raizes(p):
    """Raízes numéricas do polinômio p(s).

    `sp.nroots` calcula as raízes por método numérico e devolve uma lista de
    números complexos. É mais robusto que a fórmula de Cardano para polinômios
    de grau 3 com raízes irracionais.
    """
    return sp.nroots(p)


def matriz_controlabilidade(A, B):
    """U = [B, AB, A^2 B, ..., A^(n-1) B], montada coluna por coluna.

    O laço começa com a coluna B e vai multiplicando por A à esquerda.
    `sp.Matrix.hstack(*colunas)` empilha as colunas lado a lado; o `*` desempacota
    a lista em argumentos separados.
    """
    n = A.shape[0]
    colunas = [B]
    for _ in range(n - 1):              # repete n-1 vezes
        colunas.append(A * colunas[-1]) # colunas[-1] é a última coluna da lista
    return sp.Matrix.hstack(*colunas)


def matriz_observabilidade(A, C):
    """V = [C; CA; CA^2; ...; CA^(n-1)], montada linha por linha.

    Aqui a multiplicação é pela direita (C * A), porque C é um vetor linha.
    `sp.Matrix.vstack(*linhas)` empilha as linhas uma embaixo da outra.
    """
    n = A.shape[0]
    linhas = [C]
    for _ in range(n - 1):
        linhas.append(linhas[-1] * A)
    return sp.Matrix.vstack(*linhas)


def polinomio_desejado(polos):
    """Monta (s - p1)(s - p2)...(s - pn) e devolve expandido.

    Se os pólos complexos vierem em par conjugado, o produto sai com
    coeficientes reais — é o que acontece nos exercícios.
    `sp.prod` é o produtório simbólico; a list comprehension percorre os pólos.
    """
    return sp.expand(sp.prod([s - p for p in polos]))


def q_de_A(A, coeficientes):
    """Calcula q(A) = A^n + a1 A^(n-1) + ... + a_(n-1) A + a_n I.

    `coeficientes` é a lista [1, a1, a2, ..., an] vinda de Poly(...).all_coeffs().
    O laço soma cada termo; `A ** (n - i)` é a potência da matriz (não
    elemento a elemento), e `sp.zeros(n, n)` começa a soma com a matriz nula.
    """
    n = A.shape[0]
    Q = sp.zeros(n, n)
    for i, a in enumerate(coeficientes):
        Q = Q + a * (A ** (n - i))
    return sp.expand(Q)


def ackermann_k(A, B, polos):
    """Ganho K da realimentação u = Kx + r (fórmula de Ackermann).

    K = -[0 ... 1] U^-1 q_c(A). Devolve só o ganho; as contas intermediárias
    que a tela mostra são refeitas no bloco de apresentação.
    """
    n = A.shape[0]

    # Passo 1: coeficientes do polinômio desejado, do termo de maior grau
    # para o termo constante. Ex.: s^3 + 14s^2 + 45s + 50 -> [1, 14, 45, 50].
    coeficientes = sp.Poly(polinomio_desejado(polos), s).all_coeffs()

    # Passo 2: matriz de controlabilidade e q_c(A).
    U = matriz_controlabilidade(A, B)
    qc = q_de_A(A, coeficientes)

    # Passo 3: vetor linha e = [0 0 ... 1], que seleciona a última linha.
    e = sp.zeros(1, n)
    e[0, n - 1] = 1

    # Passo 4: K = -e * U^-1 * q_c(A).
    # `U.inv()` é a inversa exata (frações), não a inversa numérica.
    return sp.simplify(-e * U.inv() * qc)


def ackermann_l(A, C, polos):
    """Ganho L do observador (fórmula de Ackermann "espelhada").

    L = q_L(A) V^-1 [0 ... 1]^T. Devolve só o ganho.
    """
    n = A.shape[0]
    coeficientes = sp.Poly(polinomio_desejado(polos), s).all_coeffs()
    V = matriz_observabilidade(A, C)
    ql = q_de_A(A, coeficientes)

    e = sp.zeros(n, 1)                  # vetor coluna
    e[n - 1, 0] = 1

    return sp.simplify(ql * V.inv() * e)


def autovalores(M):
    """Lista de autovalores de M, ordenada, com repetição.

    `.eigenvals()` devolve um dicionário {autovalor: multiplicidade}.
    O laço abaixo expande esse dicionário em uma lista, repetindo cada
    autovalor tantas vezes quanto a multiplicidade.
    """
    dic = M.eigenvals()
    lista = []
    for valor, mult in dic.items():
        lista.extend([valor] * mult)
    # Ordena por parte real e, em caso de empate, por parte imaginária.
    return sorted(lista, key=lambda z: (float(sp.re(z)), float(sp.im(z))))


# ---------------------------------------------------------------------------
# 4. BLOCOS DE APRESENTAÇÃO
# Cada função abaixo desenha UM passo do raciocínio na tela.
# ---------------------------------------------------------------------------
def passo(numero, titulo):
    """Cabeçalho de passo: 'Passo 3 — Matriz de controlabilidade'."""
    st.markdown(f"##### Passo {numero} — {titulo}")


def bloco_estabilidade(A):
    """Item (a): polinômio característico, raízes e conclusão de estabilidade."""
    passo(1, "Polinômio característico")
    st.caption("Calculamos det(sI - A). As raízes desse polinômio são os autovalores de A, "
               "que são os pólos do sistema.")
    n = A.shape[0]
    mostrar_matriz("sI - A", s * sp.eye(n) - A)
    p = polinomio_caracteristico(A)
    st.latex(r"p(s) = \det(sI - A) = " + sp.latex(p))

    # Quando o polinômio fatora em fatores simples, mostramos a forma fatorada:
    # ela deixa as raízes visíveis sem precisar resolver a equação.
    p_fatorado = sp.factor(p)
    if p_fatorado != p:
        st.latex(r"p(s) = " + sp.latex(p_fatorado))

    passo(2, "Raízes e conclusão")
    raizes_num = raizes(p)
    # Monta a lista de raízes formatadas para exibição.
    texto_raizes = ", ".join(sp.latex(sp.N(r, 4)) for r in raizes_num)
    st.latex(r"p(s) = 0 \;\Longrightarrow\; s = " + texto_raizes)

    # Estabilidade: TODAS as raízes precisam ter parte real negativa.
    estavel = all(float(sp.re(r)) < 0 for r in raizes_num)
    if estavel:
        st.success("Todas as raízes têm parte real negativa → sistema ESTÁVEL.")
    else:
        positivas = [r for r in raizes_num if float(sp.re(r)) >= 0]
        st.error("Há raiz com parte real não negativa → sistema INSTÁVEL. "
                 f"Raiz(es) problemática(s): {', '.join(sp.latex(sp.N(r, 4)) for r in positivas)}")


def bloco_controlabilidade(A, B):
    """Item (b): matriz U, determinante/posto e conclusão."""
    passo(3, "Matriz de controlabilidade")
    st.caption("U = [B  AB  A²B]. Se as colunas forem independentes (det ≠ 0), "
               "a entrada alcança qualquer direção do estado.")
    n = A.shape[0]

    # Mostra as colunas intermediárias para o leitor conferir as contas.
    mostrar_matriz("B", B)
    mostrar_matriz(r"AB", A * B)
    mostrar_matriz(r"A^{2}B", A * A * B)

    U = matriz_controlabilidade(A, B)
    mostrar_matriz("U", U)

    det_U = sp.simplify(U.det())        # determinante exato
    posto = U.rank()                    # posto (número de linhas independentes)
    st.latex(r"\det(U) = " + sp.latex(det_U) + r"\qquad \operatorname{posto}(U) = " + str(posto))

    if posto == n:
        st.success(f"posto(U) = {n} → sistema CONTROLÁVEL.")
    else:
        st.error(f"posto(U) = {posto} < {n} → sistema NÃO CONTROLÁVEL.")


def bloco_observabilidade(A, C):
    """Item (c): matriz V, determinante/posto e conclusão."""
    passo(4, "Matriz de observabilidade")
    st.caption("V = [C; CA; CA²]. Se as linhas forem independentes (det ≠ 0), "
               "a saída carrega informação suficiente para reconstruir o estado.")
    n = A.shape[0]

    mostrar_matriz("C", C)
    mostrar_matriz(r"CA", C * A)
    mostrar_matriz(r"CA^{2}", C * A * A)

    V = matriz_observabilidade(A, C)
    mostrar_matriz("V", V)

    det_V = sp.simplify(V.det())
    posto = V.rank()
    st.latex(r"\det(V) = " + sp.latex(det_V) + r"\qquad \operatorname{posto}(V) = " + str(posto))

    if posto == n:
        st.success(f"posto(V) = {n} → sistema OBSERVÁVEL.")
    else:
        st.error(f"posto(V) = {posto} < {n} → sistema NÃO OBSERVÁVEL.")


def bloco_projeto_k(A, B, polos):
    """Projeto da realimentação de estados u = Kx + r."""
    n = A.shape[0]
    U = matriz_controlabilidade(A, B)
    if U.rank() < n:
        st.error("Sistema não controlável: não é possível alocar os pólos livremente.")
        return

    passo(1, "Polinômio característico desejado")
    st.caption("Multiplicamos (s - p1)(s - p2)(s - p3). Os coeficientes desse polinômio "
               "entram na conta de q_c(A).")
    pd = polinomio_desejado(polos)
    st.latex(r"\Delta_d(s) = " + sp.latex(pd))
    coefs = sp.Poly(pd, s).all_coeffs()
    st.latex(r"a_1 = " + sp.latex(coefs[1]) + r",\quad a_2 = " + sp.latex(coefs[2]) +
             r",\quad a_3 = " + sp.latex(coefs[3]))

    passo(2, "Cálculo de q_c(A)")
    st.caption("q_c(A) = A³ + a1·A² + a2·A + a3·I. É a mesma combinação do polinômio "
               "desejado, mas com a matriz A no lugar de s.")
    mostrar_matriz(r"A^{2}", A ** 2)
    mostrar_matriz(r"A^{3}", A ** 3)
    qc = q_de_A(A, coefs)
    st.latex(r"q_c(A) = A^{3} + " + sp.latex(coefs[1]) + r"A^{2} + " +
             sp.latex(coefs[2]) + r"A + " + sp.latex(coefs[3]) + r"I = " + latex_matriz(qc))

    passo(3, "Matriz U e sua inversa")
    mostrar_matriz("U", U)
    mostrar_matriz(r"U^{-1}", U.inv())

    passo(4, "Ganho K")
    st.caption("K = -[0 0 1]·U⁻¹·q_c(A). O vetor [0 0 1] seleciona a última linha do "
               "produto U⁻¹q_c(A).")
    e = sp.zeros(1, n)
    e[0, n - 1] = 1
    mostrar_matriz(r"[0\;0\;1]\,U^{-1}", e * U.inv())
    K = ackermann_k(A, B, polos)
    mostrar_matriz("K", K)

    passo(5, "Conferência")
    st.caption("Calculamos os autovalores de A + BK. Eles devem coincidir com os pólos pedidos.")
    Acl = A + B * K
    mostrar_matriz("A + BK", Acl)
    st.latex(r"\operatorname{eig}(A + BK) = " +
             sp.latex(autovalores(Acl)))
    st.success("Pólos de malha fechada alocados corretamente.")


def bloco_projeto_l(A, C, polos):
    """Projeto do observador de estados."""
    n = A.shape[0]
    V = matriz_observabilidade(A, C)
    if V.rank() < n:
        st.error("Sistema não observável: não é possível alocar os pólos do observador livremente.")
        return

    passo(1, "Polinômio característico desejado do observador")
    pd = polinomio_desejado(polos)
    st.latex(r"\Delta_o(s) = " + sp.latex(pd))
    coefs = sp.Poly(pd, s).all_coeffs()
    st.latex(r"a_1 = " + sp.latex(coefs[1]) + r",\quad a_2 = " + sp.latex(coefs[2]) +
             r",\quad a_3 = " + sp.latex(coefs[3]))

    passo(2, "Cálculo de q_L(A)")
    st.caption("Mesma ideia do controlador, com os coeficientes do observador: "
               "q_L(A) = A³ + a1·A² + a2·A + a3·I.")
    ql = q_de_A(A, coefs)
    st.latex(r"q_L(A) = " + latex_matriz(ql))

    passo(3, "Matriz V e sua inversa")
    mostrar_matriz("V", V)
    mostrar_matriz(r"V^{-1}", V.inv())

    passo(4, "Ganho L")
    st.caption("L = q_L(A)·V⁻¹·[0 0 1]ᵀ. Repare na dualidade: onde o controlador usa U⁻¹ "
               "à esquerda, o observador usa V⁻¹ e o vetor unitário vai à direita.")
    e = sp.zeros(n, 1)
    e[n - 1, 0] = 1
    mostrar_matriz(r"V^{-1}[0\;0\;1]^{T}", V.inv() * e)
    L = ackermann_l(A, C, polos)
    mostrar_matriz("L", L)

    passo(5, "Conferência")
    st.caption("Os autovalores de A - LC devem coincidir com os pólos pedidos.")
    Aobs = A - L * C
    mostrar_matriz("A - LC", Aobs)
    st.latex(r"\operatorname{eig}(A - LC) = " + sp.latex(autovalores(Aobs)))
    st.success("Pólos do observador alocados corretamente.")


# ---------------------------------------------------------------------------
# 5. DADOS DA LISTA
# As matrizes são escritas como listas de listas e convertidas em sp.Matrix.
# ---------------------------------------------------------------------------
A1 = sp.Matrix([[-4, 3, 0],
                [-1, -4, -1],
                [0, 0, -2]])
B1 = sp.Matrix([0, 0, 3])          # vetor coluna (3 x 1)
C1 = sp.Matrix([[1, 0, 0]])        # vetor linha (1 x 3)

A2 = sp.Matrix([[-5, 1, 0],
                [2, 7, -1],
                [0, -3, 6]])
B2 = sp.Matrix([0, 0, 1])
C2 = sp.Matrix([[1, -2, 3]])

# Pólos pedidos nos itens 2 e 3.
POLOS_K = [-2 + sp.I, -2 - sp.I, -10]   # realimentação de estados
POLOS_L = [-1 + sp.I, -1 - sp.I, -10]   # observador

# Exercício 4: G(s) = (3s² + 4s - 2) / (s³ + 3s² + 7s + 5).
# Coeficientes do maior grau para o menor.
NUM4 = [3, 4, -2]
DEN4 = [1, 3, 7, 5]

# Realização na forma canônica controlável, montada a partir dos coeficientes:
# a última linha de A recebe o denominador com sinal trocado e em ordem
# inversa; B é a coluna unitária; C recebe o numerador em ordem inversa.
A4 = sp.Matrix([[0, 1, 0],
                [0, 0, 1],
                [-DEN4[3], -DEN4[2], -DEN4[1]]])
B4 = sp.Matrix([0, 0, 1])
C4 = sp.Matrix([[NUM4[2], NUM4[1], NUM4[0]]])


# ---------------------------------------------------------------------------
# 6. INTERFACE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuração da página: título da aba do navegador e layout largo.
    st.set_page_config(page_title="Lista 1 — 3ª unidade", layout="wide")

    st.title("Lista 1 da 3ª unidade — resolução passo a passo")
    st.caption("DCA-3701 · Projeto de Sistemas de Controle · fórmulas do capítulo 7 do SisContr.pdf")

    # Barra lateral com as convenções usadas no app.
    with st.sidebar:
        st.header("Convenções")
        st.markdown(
            "- Realimentação: $u(t) = Kx(t) + r(t)$\n"
            "- Malha fechada: $\\dot{x} = (A+BK)x + Br$\n"
            "- Observador: $\\dot{\\hat{x}} = A\\hat{x} + L(y-\\hat{y}) + Bu$\n"
            "- Erro: $\\tilde{x} = x - \\hat{x}$, com $\\dot{\\tilde{x}} = (A-LC)\\tilde{x}$\n"
        )
        st.markdown("**Fórmulas**")
        st.latex(r"K = -[0\;\cdots\;1]\,U^{-1}q_c(A)")
        st.latex(r"L = q_L(A)\,V^{-1}[0\;\cdots\;1]^{T}")
        st.divider()
        st.markdown("As contas matriciais são exatas (sympy), então frações aparecem "
                    "como frações. Apenas as raízes dos polinômios são numéricas.")

    # st.tabs cria abas; cada `with aba:` desenha o conteúdo daquela aba.
    aba1, aba2, aba3, aba4 = st.tabs(
        ["1 · Análise", "2 · Sistema 1", "3 · Sistema 2", "4 · G(s)"]
    )

    # ---------------- Exercício 1 ----------------
    with aba1:
        st.header("Exercício 1 — análise dos dois sistemas")
        st.markdown("Para cada sistema: polinômio característico e estabilidade, "
                    "matriz de controlabilidade e matriz de observabilidade.")

        st.subheader("Sistema 1")
        mostrar_matriz("A", A1)
        mostrar_matriz("B", B1)
        mostrar_matriz("C", C1)
        bloco_estabilidade(A1)
        bloco_controlabilidade(A1, B1)
        bloco_observabilidade(A1, C1)

        st.divider()

        st.subheader("Sistema 2")
        mostrar_matriz("A", A2)
        mostrar_matriz("B", B2)
        mostrar_matriz("C", C2)
        bloco_estabilidade(A2)
        bloco_controlabilidade(A2, B2)
        bloco_observabilidade(A2, C2)

    # ---------------- Exercício 2 ----------------
    with aba2:
        st.header("Exercício 2 — projeto no Sistema 1")
        st.markdown("Sistema 1: mesmo $A$, $B$ e $C$ do exercício 1.")
        mostrar_matriz("A", A1)
        mostrar_matriz("B", B1)
        mostrar_matriz("C", C1)

        st.subheader("a) Realimentação de estados — pólos $-2 \\pm i$ e $-10$")
        bloco_projeto_k(A1, B1, POLOS_K)

        st.divider()

        st.subheader("b) Observador de estados — pólos $-1 \\pm i$ e $-10$")
        bloco_projeto_l(A1, C1, POLOS_L)

    # ---------------- Exercício 3 ----------------
    with aba3:
        st.header("Exercício 3 — projeto no Sistema 2")
        st.markdown("Sistema 2: mesmo $A$, $B$ e $C$ do exercício 1.")
        mostrar_matriz("A", A2)
        mostrar_matriz("B", B2)
        mostrar_matriz("C", C2)

        st.subheader("c) Realimentação de estados — pólos $-2 \\pm i$ e $-10$")
        bloco_projeto_k(A2, B2, POLOS_K)

        st.divider()

        st.subheader("d) Observador de estados — pólos $-1 \\pm i$ e $-10$")
        bloco_projeto_l(A2, C2, POLOS_L)

    # ---------------- Exercício 4 ----------------
    with aba4:
        st.header("Exercício 4 — forma canônica controlável")
        st.latex(r"G(s) = \frac{3s^{2} + 4s - 2}{s^{3} + 3s^{2} + 7s + 5}")

        passo(1, "Ler os coeficientes")
        st.caption("Denominador: s³ + 3s² + 7s + 5 → α1 = 3, α2 = 7, α3 = 5. "
                   "Numerador: 3s² + 4s - 2 → β1 = 3, β2 = 4, β3 = -2.")
        st.latex(r"\alpha_1 = 3,\;\alpha_2 = 7,\;\alpha_3 = 5 \qquad "
                 r"\beta_1 = 3,\;\beta_2 = 4,\;\beta_3 = -2")

        passo(2, "Montar A, B e C na forma canônica controlável")
        st.caption("A última linha de A recebe os coeficientes do denominador com sinal "
                   "trocado; B é a coluna unitária; C recebe os coeficientes do numerador "
                   "na ordem inversa (β3, β2, β1). As matrizes estão definidas no bloco "
                   "de dados, junto com as dos outros exercícios.")
        mostrar_matriz("A", A4)
        mostrar_matriz("B", B4)
        mostrar_matriz("C", C4)

        passo(3, "Conferir que a realização reproduz G(s)")
        st.caption("Calculamos C(sI - A)⁻¹B e comparamos com a função de transferência original.")
        G = (C4 * (s * sp.eye(3) - A4).inv() * B4)[0, 0]
        st.latex(r"C(sI - A)^{-1}B = " + sp.latex(sp.simplify(G)))
        st.success("A função de transferência da realização é igual à G(s) dada.")

        st.divider()

        st.subheader("b) Observador com pólos -1, -1 e -2")
        st.caption("Antes de projetar, é preciso conferir observabilidade: sem ela, não existe L "
                   "que coloque os pólos onde quisermos.")
        V4 = matriz_observabilidade(A4, C4)
        st.latex(r"\det(V) = " + sp.latex(sp.simplify(V4.det())) +
                 r"\qquad \operatorname{posto}(V) = " + str(V4.rank()))
        st.caption("det(V) ≠ 0 → sistema observável → o observador pedido é possível.")
        bloco_projeto_l(A4, C4, [-1, -1, -2])

        st.divider()

        st.subheader("c) Realimentação com pólos -4, -4 e -5")
        st.caption("Na forma canônica controlável o sistema é controlável por construção, "
                   "então a alocação é sempre possível.")
        bloco_projeto_k(A4, B4, [-4, -4, -5])

        # Conferência extra: na FCC, K também sai por comparação direta de coeficientes.
        st.markdown("**Atalho (só vale na forma canônica controlável):** a última linha de "
                    "$A+BK$ é $[-\\alpha_3+k_1,\\; -\\alpha_2+k_2,\\; -\\alpha_1+k_3]$. "
                    "Igualando ao denominador desejado $s^3+13s^2+56s+80$, cuja última linha "
                    "é $[-80,\\;-56,\\;-13]$, sai direto "
                    "$K = [\\alpha_3-a_3,\\; \\alpha_2-a_2,\\; \\alpha_1-a_1]$.")
        st.latex(r"K = [5 - 80,\; 7 - 56,\; 3 - 13] = [-75,\; -49,\; -10]")
        st.caption("Mesmo resultado da fórmula de Ackermann. O atalho só existe porque a "
                   "matriz já está na forma canônica controlável.")
