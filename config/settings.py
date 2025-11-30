import os
from pathlib import Path

# Carregar variáveis de ambiente do .env (para desenvolvimento local)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

class Settings:
    """
    Configurações com suporte a Streamlit Secrets e variáveis de ambiente.
    Os secrets são lidos dinamicamente (não no import time).
    """
    
    @staticmethod
    def _get(key: str, default: str = "") -> str:
        """Obtém valor do Streamlit secrets ou env vars"""
        # 1. Tentar Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
        
        # 2. Fallback para variáveis de ambiente
        return os.getenv(key, default)
    
    # === API Keys (propriedades dinâmicas) ===
    @property
    def GOOGLE_API_KEY(self) -> str:
        return self._get("GOOGLE_API_KEY", "")
    
    @property
    def OPENROUTER_API_KEY(self) -> str:
        return self._get("OPENROUTER_API_KEY", "")
    
    @property
    def MISTRAL_API_KEY(self) -> str:
        return self._get("MISTRAL_API_KEY", "")
    
    @property
    def HUGGINGFACE_API_KEY(self) -> str:
        return self._get("HUGGINGFACE_API_KEY", "")
    
    @property
    def TOGETHER_API_KEY(self) -> str:
        return self._get("TOGETHER_API_KEY", "")
    
    # === VLM Models ===
    @property
    def VLM_MODEL(self) -> str:
        return self._get("VLM_MODEL", "gemini-2.5-flash")
    
    @property
    def VLM_BACKUP(self) -> str:
        return self._get("VLM_BACKUP", "gemini-2.5-flash-lite")
    
    @property
    def VLM_BACKUP_2(self) -> str:
        return self._get("VLM_BACKUP_2", "pixtral-12b-2409")
    
    # === LLM Models ===
    @property
    def LLM_MODEL(self) -> str:
        return self._get("LLM_MODEL", "gemini-2.5-pro")
    
    @property
    def LLM_BACKUP(self) -> str:
        return self._get("LLM_BACKUP", "mistral-small-latest")
    
    @property
    def LLM_OPENROUTER_1(self) -> str:
        return self._get("LLM_OPENROUTER_1", "x-ai/grok-4.1-fast:free")
    
    @property
    def LLM_OPENROUTER_2(self) -> str:
        return self._get("LLM_OPENROUTER_2", "tngtech/deepseek-r1t-chimera:free")
    
    @property
    def LLM_OPENROUTER_3(self) -> str:
        return self._get("LLM_OPENROUTER_3", "google/gemma-3-27b-it:free")
    
    @property
    def LLM_OPENROUTER_4(self) -> str:
        return self._get("LLM_OPENROUTER_4", "z-ai/glm-4.5-air:free")
    
    # === Embeddings ===
    @property
    def EMBEDDING_MODEL(self) -> str:
        return self._get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    @property
    def GOOGLE_EMBEDDING_MODEL(self) -> str:
        return self._get("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
    
    # === Paths ===
    @property
    def CHROMA_PATH(self) -> str:
        return self._get("CHROMA_PATH", "./knowledge_base/chroma_db")
    
    @property
    def DOCS_PATH(self) -> str:
        return self._get("DOCS_PATH", "./knowledge_base/documents")


# Instância global
settings = Settings()
