import os
from dataclasses import dataclass
from pathlib import Path

# Carregar variáveis de ambiente do .env
try:
    from dotenv import load_dotenv
    # Procurar .env na raiz do projeto
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv não instalado

# Função para obter secrets (Streamlit Cloud ou env vars)
def get_secret(key: str, default: str = "") -> str:
    """Obtém secret do Streamlit Cloud ou variáveis de ambiente"""
    # Tentar Streamlit secrets primeiro
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    # Fallback para variáveis de ambiente
    return os.getenv(key, default)

@dataclass
class Settings:
    # API Keys
    OPENROUTER_API_KEY: str = get_secret("OPENROUTER_API_KEY", "")
    GOOGLE_API_KEY: str = get_secret("GOOGLE_API_KEY", "")
    MISTRAL_API_KEY: str = get_secret("MISTRAL_API_KEY", "")
    HUGGINGFACE_API_KEY: str = get_secret("HUGGINGFACE_API_KEY", "")
    TOGETHER_API_KEY: str = get_secret("TOGETHER_API_KEY", "")
    
    # VLM (Vision Language Model) - Análise de Imagens
    VLM_MODEL: str = get_secret("VLM_MODEL", "gemini-2.5-flash")
    VLM_BACKUP: str = get_secret("VLM_BACKUP", "qwen/qwen2.5-vl-72b-instruct:free")
    VLM_BACKUP_2: str = get_secret("VLM_BACKUP_2", "mistralai/pixtral-12b:free")
    VLM_BACKUP_3: str = get_secret("VLM_BACKUP_3", "gemini-2.5-flash-lite")
    
    # LLM (Large Language Model) - Raciocínio Clínico
    LLM_MODEL: str = get_secret("LLM_MODEL", "gemini-2.5-pro")
    LLM_BACKUP: str = get_secret("LLM_BACKUP", "mistral-small-latest")
    
    # OpenRouter Free Models (Novembro 2025)
    LLM_OPENROUTER_1: str = get_secret("LLM_OPENROUTER_1", "x-ai/grok-4.1-fast:free")
    LLM_OPENROUTER_2: str = get_secret("LLM_OPENROUTER_2", "tngtech/deepseek-r1t-chimera:free")
    LLM_OPENROUTER_3: str = get_secret("LLM_OPENROUTER_3", "google/gemma-3-27b-it:free")
    LLM_OPENROUTER_4: str = get_secret("LLM_OPENROUTER_4", "z-ai/glm-4.5-air:free")
    
    # Embeddings (Pesquisa Semântica) - HuggingFace local
    EMBEDDING_MODEL: str = get_secret("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Google embeddings (requer API diferente)
    GOOGLE_EMBEDDING_MODEL: str = get_secret("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
    
    # Paths
    CHROMA_PATH: str = get_secret("CHROMA_PATH", "./knowledge_base/chroma_db")
    DOCS_PATH: str = get_secret("DOCS_PATH", "./knowledge_base/documents")

settings = Settings()
