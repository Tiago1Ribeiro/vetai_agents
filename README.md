# 🐾 Vet Agents

Sistema inteligente de apoio ao diagnóstico veterinário com IA.

## Funcionalidades

- 🔍 **Análise de Imagens** - Gemini Vision / Pixtral para análise visual
- 🧠 **Diagnóstico com IA** - Múltiplos LLMs (Grok, Gemma, DeepSeek)
- 📚 **Pesquisa Web** - Informação veterinária actualizada
- 🗄️ **RAG Local** - Base de conhecimento própria

## ⚠️ Aviso Importante

Este sistema é uma **ferramenta de apoio** e **NÃO substitui** a consulta presencial com um médico veterinário qualificado.

## Deploy no Streamlit Cloud

1. Fork este repositório
2. Vai a [share.streamlit.io](https://share.streamlit.io)
3. Conecta o teu GitHub
4. Seleciona o repositório e `app.py`
5. Configura os **Secrets** (ver abaixo)

### Configurar Secrets

No dashboard do Streamlit Cloud, adiciona estes secrets:

```toml
GOOGLE_API_KEY = "sua-chave-google"
OPENROUTER_API_KEY = "sua-chave-openrouter"
MISTRAL_API_KEY = "sua-chave-mistral"
```

## Executar Localmente

```bash
# Instalar dependências
pip install -r requirements.txt

# Criar .env com as API keys
cp .streamlit/secrets.toml.example .env

# Executar
streamlit run app.py
```

## Tecnologias

- **Frontend**: Streamlit
- **Vision AI**: Google Gemini, Mistral Pixtral
- **LLM**: OpenRouter (Grok, Gemma, DeepSeek), Mistral
- **RAG**: ChromaDB + HuggingFace Embeddings
- **Web Search**: DuckDuckGo

## Licença

MIT
