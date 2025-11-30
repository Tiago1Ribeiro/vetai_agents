#  Vet Agents

Sistema multi-agente de apoio ao diagnóstico veterinário com IA generativa.

** Demo: [vetai-agents.streamlit.app](https://vetai-agents.streamlit.app)**

##  O que é

Ferramenta de IA para apoio a médicos veterinários na análise de casos clínicos. Combina análise de imagem, pesquisa web e geração de diagnósticos diferenciais usando modelos de linguagem.

##  Arquitetura

`

                      Streamlit UI                           

                    Orchestrator Agent                       
         (coordena o fluxo entre os agentes)                 

 Vision Agent  Knowledge      Web Search   Diagnosis     
 (imagens)     Agent (RAG)    Tool         Agent (LLM)   

 Gemini 2.5    ChromaDB +    DuckDuckGo    DeepSeek R1   
 Flash         MiniLM-L6     (ddgs)        Chimera       

`

##  Modelos (100% Gratuitos)

### Visão (Análise de Imagens)
| Modelo | Provider | Uso |
|--------|----------|-----|
| Gemini 2.5 Flash | Google AI | Análise principal |
| Gemini 2.5 Flash Lite | Google AI | Análise rápida |
| Pixtral 12B | Mistral | Alternativa |

### Texto (Diagnóstico)
| Modelo | Provider | Uso |
|--------|----------|-----|
| DeepSeek R1 Chimera | OpenRouter | Default - raciocínio |
| Grok 4.1 Fast | OpenRouter | Alternativa rápida |
| Gemma 3 27B | OpenRouter | Google open-source |
| GLM 4.5 Air | OpenRouter | Chinês multilingual |

### RAG & Embeddings
- **ChromaDB** - Vector store local
- **all-MiniLM-L6-v2** - Embeddings (HuggingFace)

##  Funcionalidades

-  **Upload de imagens** clínicas (lesões, radiografias, etc.)
-  **Análise visual** com Gemini Vision
-  **Pesquisa web** automática de literatura veterinária
-  **RAG** com base de conhecimento local
-  **Diagnóstico diferencial** estruturado
-  **100% gratuito** - usa apenas APIs free tier

##  Deploy

Deployed no **Streamlit Community Cloud**:
1. Push para GitHub
2. Conectar em [share.streamlit.io](https://share.streamlit.io)
3. Configurar Secrets (API keys)
4. Deploy automático

### Secrets necessários

`	oml
GOOGLE_API_KEY = "AIzaSy..."      # Google AI Studio
OPENROUTER_API_KEY = "sk-or-..." # OpenRouter.ai
MISTRAL_API_KEY = "..."          # Mistral (opcional)
`

##  Executar Localmente

`ash
# Clone
git clone https://github.com/Tiago1Ribeiro/vetai_agents.git
cd vetai_agents

# Ambiente
conda create -n vet_agents python=3.11
conda activate vet_agents
pip install -r requirements.txt

# Configurar keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Editar com as tuas API keys

# Run
streamlit run app.py
`

##  Estrutura

`
 app.py                 # Interface Streamlit
 agents/
    orchestrator.py    # Coordenador principal
    vision_agent.py    # Análise de imagens
    knowledge_agent.py # RAG + Web search
    diagnosis_agent.py # Geração de diagnóstico
 tools/
    web_search.py      # RAG tool
    web_search_tool.py # DuckDuckGo search
 config/
    settings.py        # Configurações e secrets
 requirements.txt
`

##  Aviso Importante

> Este sistema é uma **ferramenta de apoio** e **NÃO substitui** a consulta presencial com um médico veterinário qualificado.

##  Licença

MIT

---
*Desenvolvido como projeto de exploração de arquiteturas multi-agente com LLMs.*
