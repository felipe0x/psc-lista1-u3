# Lista 1 da 3ª unidade — DCA-3701

App Streamlit que resolve, passo a passo, os quatro exercícios da lista de Projeto de
Sistemas de Controle (UFRN): polinômio característico e estabilidade, matrizes de
controlabilidade e observabilidade, realimentação de estados (Ackermann) e observador
de estados.

## Rodar localmente

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app_lista1.py
```

Abre em `http://localhost:8501`.

## Deploy

Streamlit Community Cloud: <https://share.streamlit.io> → *New app* → repositório,
branch `main`, arquivo principal `app_lista1.py`.

## Conteúdo

- `app_lista1.py` — aplicação (quatro abas, uma por exercício).
- `requirements.txt` — `streamlit` e `sympy`.
