"""
Ferramenta RAG (Retrieval-Augmented Generation) para conhecimento veterinário.

Funcionalidades:
- Ingestão de documentos (PDF, TXT, MD)
- Pesquisa semântica com ChromaDB
- Reranking de resultados
- Cache de embeddings
"""

# Suprimir warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangChain.*")
warnings.filterwarnings("ignore", message=".*Chroma.*")

import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
import json

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document


@dataclass
class SearchResult:
    """Resultado de uma pesquisa RAG"""
    content: str
    source: str
    page: Optional[int]
    score: float
    metadata: dict


@dataclass
class RAGConfig:
    """Configuração do sistema RAG"""
    chroma_path: str = "./knowledge_base/chroma_db"
    docs_path: str = "./knowledge_base/documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "veterinary_docs"


class RAGTool:
    """
    Ferramenta RAG para consulta de documentos veterinários.
    
    Usa ChromaDB como vector store e HuggingFace embeddings (local).
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Inicializa o RAGTool.
        
        Args:
            config: Configuração personalizada (opcional)
        """
        self.config = config or RAGConfig()
        
        # Criar diretórios se não existirem
        Path(self.config.chroma_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.docs_path).mkdir(parents=True, exist_ok=True)
        
        # Inicializar embeddings (corre localmente)
        print(f"📦 Carregando modelo de embeddings: {self.config.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Inicializar/carregar vector store
        self.vectorstore = Chroma(
            collection_name=self.config.collection_name,
            persist_directory=self.config.chroma_path,
            embedding_function=self.embeddings
        )
        
        # Text splitter para ingestão
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Cache de documentos ingeridos
        self.ingested_cache_path = Path(self.config.chroma_path) / "ingested_files.json"
        self.ingested_files = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Carrega cache de ficheiros ingeridos"""
        if self.ingested_cache_path.exists():
            with open(self.ingested_cache_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Guarda cache de ficheiros ingeridos"""
        with open(self.ingested_cache_path, 'w') as f:
            json.dump(self.ingested_files, f, indent=2)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Calcula hash de um ficheiro"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_loader(self, file_path: Path):
        """Retorna o loader apropriado para o tipo de ficheiro"""
        suffix = file_path.suffix.lower()
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
        }
        
        loader_class = loaders.get(suffix)
        if loader_class:
            return loader_class(str(file_path))
        
        raise ValueError(f"Tipo de ficheiro não suportado: {suffix}")
    
    def ingest_file(
        self,
        file_path: Union[str, Path],
        force: bool = False
    ) -> int:
        """
        Ingere um único ficheiro.
        
        Args:
            file_path: Caminho do ficheiro
            force: Força reingestão mesmo se já existe
            
        Returns:
            Número de chunks adicionados
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Ficheiro não encontrado: {path}")
        
        # Verificar se já foi ingerido
        file_hash = self._compute_file_hash(path)
        cache_key = str(path.absolute())
        
        if not force and cache_key in self.ingested_files:
            if self.ingested_files[cache_key]['hash'] == file_hash:
                print(f"⏭️ Ficheiro já ingerido: {path.name}")
                return 0
        
        print(f"📄 Ingerindo: {path.name}")
        
        # Carregar documento
        loader = self._get_loader(path)
        documents = loader.load()
        
        # Adicionar metadados
        for doc in documents:
            doc.metadata['source'] = str(path.name)
            doc.metadata['full_path'] = str(path.absolute())
            doc.metadata['file_type'] = path.suffix
        
        # Dividir em chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Adicionar ao vector store
        if chunks:
            self.vectorstore.add_documents(chunks)
            
            # Atualizar cache
            self.ingested_files[cache_key] = {
                'hash': file_hash,
                'chunks': len(chunks),
                'name': path.name
            }
            self._save_cache()
        
        print(f"   ✅ Adicionados {len(chunks)} chunks")
        return len(chunks)
    
    def ingest_directory(
        self,
        directory: Optional[Union[str, Path]] = None,
        extensions: list[str] = None,
        force: bool = False
    ) -> dict:
        """
        Ingere todos os documentos de um diretório.
        
        Args:
            directory: Diretório (usa config.docs_path se None)
            extensions: Lista de extensões a processar
            force: Força reingestão
            
        Returns:
            Estatísticas da ingestão
        """
        dir_path = Path(directory or self.config.docs_path)
        extensions = extensions or ['.pdf', '.txt', '.md']
        
        stats = {'files_processed': 0, 'chunks_added': 0, 'errors': []}
        
        print(f"📁 Ingerindo documentos de: {dir_path}")
        
        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                try:
                    chunks = self.ingest_file(file_path, force=force)
                    stats['files_processed'] += 1
                    stats['chunks_added'] += chunks
                except Exception as e:
                    stats['errors'].append({'file': str(file_path), 'error': str(e)})
                    print(f"   ❌ Erro em {file_path.name}: {e}")
        
        print(f"\n📊 Resumo: {stats['files_processed']} ficheiros, {stats['chunks_added']} chunks")
        return stats
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
        min_score: float = 0.0
    ) -> list[SearchResult]:
        """
        Pesquisa semântica nos documentos.
        
        Args:
            query: Texto da pesquisa
            k: Número de resultados
            filter_dict: Filtros de metadados
            min_score: Score mínimo (0-1)
            
        Returns:
            Lista de SearchResult
        """
        # Pesquisar com scores
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filter_dict
        )
        
        # Converter para SearchResult
        search_results = []
        for doc, score in results:
            if score >= min_score:
                search_results.append(SearchResult(
                    content=doc.page_content,
                    source=doc.metadata.get('source', 'unknown'),
                    page=doc.metadata.get('page'),
                    score=float(score),
                    metadata=doc.metadata
                ))
        
        return search_results
    
    def search_with_context(
        self,
        query: str,
        k: int = 5,
        context_window: int = 1
    ) -> list[SearchResult]:
        """
        Pesquisa com contexto expandido (chunks adjacentes).
        
        Args:
            query: Texto da pesquisa
            k: Número de resultados
            context_window: Número de chunks adjacentes a incluir
        """
        # Implementação básica - pesquisa normal
        # Para contexto expandido seria necessário tracking de chunk IDs
        return self.search(query, k=k)
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        keyword_weight: float = 0.3
    ) -> list[SearchResult]:
        """
        Pesquisa híbrida: semântica + keyword.
        
        Combina similaridade de embeddings com match de keywords.
        """
        # Pesquisa semântica
        semantic_results = self.search(query, k=k*2)
        
        # Boost baseado em keywords
        query_terms = set(query.lower().split())
        
        for result in semantic_results:
            content_terms = set(result.content.lower().split())
            keyword_overlap = len(query_terms & content_terms) / len(query_terms) if query_terms else 0
            
            # Combinar scores
            result.score = (1 - keyword_weight) * result.score + keyword_weight * keyword_overlap
        
        # Reordenar e limitar
        semantic_results.sort(key=lambda x: x.score, reverse=True)
        return semantic_results[:k]
    
    def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000,
        k: int = 10
    ) -> str:
        """
        Obtém contexto relevante formatado para LLM.
        
        Args:
            query: Texto da pesquisa
            max_tokens: Limite aproximado de tokens
            k: Número máximo de chunks
            
        Returns:
            Texto formatado com contexto relevante
        """
        results = self.hybrid_search(query, k=k)
        
        context_parts = []
        current_length = 0
        char_per_token = 4  # Estimativa
        max_chars = max_tokens * char_per_token
        
        for result in results:
            chunk_text = f"[Fonte: {result.source}"
            if result.page:
                chunk_text += f", Página {result.page}"
            chunk_text += f"]\n{result.content}\n"
            
            if current_length + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n---\n".join(context_parts)
    
    def add_document(
        self,
        content: str,
        metadata: Optional[dict] = None,
        source: str = "manual"
    ):
        """
        Adiciona documento diretamente (sem ficheiro).
        
        Args:
            content: Conteúdo do documento
            metadata: Metadados adicionais
            source: Nome da fonte
        """
        metadata = metadata or {}
        metadata['source'] = source
        
        doc = Document(page_content=content, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        
        self.vectorstore.add_documents(chunks)
        print(f"✅ Adicionado documento manual: {len(chunks)} chunks")
    
    def delete_by_source(self, source: str):
        """Remove documentos de uma fonte específica"""
        # ChromaDB não suporta delete por metadados diretamente
        # Seria necessário implementar com IDs
        print(f"⚠️ Delete por fonte não implementado: {source}")
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do vector store"""
        collection = self.vectorstore._collection
        
        return {
            'total_documents': collection.count(),
            'ingested_files': len(self.ingested_files),
            'files': list(self.ingested_files.keys()),
            'embedding_model': self.config.embedding_model,
            'chunk_size': self.config.chunk_size
        }
    
    def clear(self, confirm: bool = False):
        """Limpa todo o vector store"""
        if not confirm:
            print("⚠️ Use clear(confirm=True) para confirmar")
            return
        
        self.vectorstore.delete_collection()
        self.ingested_files = {}
        self._save_cache()
        
        # Reinicializar
        self.vectorstore = Chroma(
            collection_name=self.config.collection_name,
            persist_directory=self.config.chroma_path,
            embedding_function=self.embeddings
        )
        
        print("🗑️ Vector store limpo")


# Função utilitária para uso rápido
def quick_search(query: str, k: int = 5) -> list[SearchResult]:
    """Pesquisa rápida com configuração padrão"""
    rag = RAGTool()
    return rag.search(query, k=k)


def quick_ingest(docs_path: str):
    """Ingestão rápida de um diretório"""
    rag = RAGTool(RAGConfig(docs_path=docs_path))
    return rag.ingest_directory()

