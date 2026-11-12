# Listas 1 e 2 da 3ª unidade — DCA-3701

Apps Streamlit que resolvem, passo a passo, os exercícios das listas de Projeto de
Sistemas de Controle (UFRN).

- `app_lista1.py` — polinômio característico e estabilidade, matrizes de
  controlabilidade e observabilidade, realimentação de estados (Ackermann) e
  observador de estados.
- `app_lista2.py` — formas canônicas controlável e observável e projeto do seguidor
  de referência para entrada degrau. Importa as funções genéricas de `app_lista1.py`,
  então os dois arquivos precisam ficar na mesma pasta.

## Rodar localmente

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app_lista1.py   # ou app_lista2.py
```

Abre em `http://localhost:8501`.

## Deploy

Streamlit Community Cloud: <https://share.streamlit.io> → *New app* → repositório,
branch `main`, arquivo principal `app_lista1.py` (ou `app_lista2.py`).

## Conteúdo

- `app_lista1.py` — aplicação da Lista 1 (quatro abas, uma por exercício).
- `app_lista2.py` — aplicação da Lista 2 (quatro abas, uma por exercício).
- `requirements.txt` — `streamlit` e `sympy`.
