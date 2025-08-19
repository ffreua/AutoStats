# Auto Stats App

Aplicativo web (Streamlit) para upload de planilhas CSV/Excel, com:
- Detecção automática de tipos de variáveis
- Estatística descritiva
- Sugestões de análises inferenciais (t-test, ANOVA, qui-quadrado, correlações)
- Geração de gráficos com Matplotlib
- Exportação de relatório em PDF
- Botão **Sobre** com a frase: *"Desenvolvido por Dr Fernando Freua. Copyright 20205 - Todos os direitos reservados"*

## Como executar localmente
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Acesse o endereço mostrado no terminal (ex.: http://localhost:8501).

## Como gerar um executável (Windows)
```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --add-data "app.py;." -n AutoStatsApp -F -c -i NONE -w -y app.py
```
> Dica: Executáveis de apps Streamlit são possíveis via **`streamlit run`** embalado com scripts; para distribuição, o mais estável é criar um instalador que execute `pip install -r requirements.txt` e rode `streamlit run app.py` por um atalho.

## Deploy gratuito
- **Streamlit Community Cloud**: crie um repositório no GitHub com estes arquivos e conecte em https://streamlit.io/cloud.
- **Railway/Render/Fly.io**: suporte a apps Python com web dynos.
- **PythonAnywhere (plano grátis)**: recomenda-se um micro-Flask; porém é possível rodar Streamlit via tarefa *always-on* nos planos pagos.

## Observação sobre "IA gratuita"
Este app usa uma **IA local baseada em heurísticas** (sem chaves e sem custo) para sugerir testes estatísticos com base nos tipos de variáveis e cardinalidade. Não envia dados para terceiros.