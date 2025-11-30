"""
Agente de Conhecimento - RAG + Web Search.

USA: tools/web_search.py (RAGTool)
USA: tools/web_search_tool.py (WebSearchTool)
"""

from typing import Optional
from config.settings import settings

# ═══════════════════════════════════════════════════════════════
# IMPORTA AS TOOLS
# ═══════════════════════════════════════════════════════════════
from tools.web_search import RAGTool, RAGConfig
from tools.web_search_tool import WebSearchTool


class KnowledgeAgent:
    """Agente de pesquisa: RAG local + Web Search"""
    
    def __init__(self):
        # ═══════════════════════════════════════════════════════
        # INICIALIZA RAGTool das tools
        # ═══════════════════════════════════════════════════════
        rag_config = RAGConfig(
            chroma_path=settings.CHROMA_PATH,
            docs_path=settings.DOCS_PATH,
            embedding_model=settings.EMBEDDING_MODEL,
            chunk_size=1000,
            chunk_overlap=200
        )
        self.rag_tool = RAGTool(config=rag_config)
        
        # ═══════════════════════════════════════════════════════
        # INICIALIZA WebSearchTool das tools
        # ═══════════════════════════════════════════════════════
        self.web_search = WebSearchTool(
            google_api_key=settings.GOOGLE_API_KEY,
            preferred_provider='gemini'  # Usa Gemini Grounding primeiro
        )
        
        print(f"📚 KnowledgeAgent inicializado")
        print(f"   - RAG: {self.rag_tool.get_stats()['total_documents']} documentos")
        print(f"   - Web: {self.web_search.get_available_providers()}")
    
    def ingest_documents(self, docs_path: Optional[str] = None) -> dict:
        """
        Ingere documentos veterinários para o RAG.
        
        Usar uma vez para carregar a base de conhecimento inicial.
        """
        # ═══════════════════════════════════════════════════════
        # USA RAGTool para ingestão
        # ═══════════════════════════════════════════════════════
        return self.rag_tool.ingest_directory(
            directory=docs_path,
            extensions=['.pdf', '.txt', '.md']
        )
    
    def search_documents(self, query: str, k: int = 5) -> list[dict]:
        """
        Pesquisa nos documentos veterinários locais.
        """
        # ═══════════════════════════════════════════════════════
        # USA RAGTool para pesquisa
        # ═══════════════════════════════════════════════════════
        results = self.rag_tool.hybrid_search(query, k=k)
        
        return [
            {
                "content": r.content,
                "source": r.source,
                "page": r.page,
                "score": r.score
            }
            for r in results
        ]
    
    def search_web(self, query: str) -> str:
        """
        Pesquisa web sobre tema veterinário.
        """
        # ═══════════════════════════════════════════════════════
        # USA WebSearchTool para pesquisa
        # ═══════════════════════════════════════════════════════
        result = self.web_search.search_veterinary(
            query,
            max_results=5,
            use_gemini_analysis=True
        )
        
        # Retorna análise do Gemini se disponível, senão formata resultados
        if result['analysis']:
            return result['analysis']
        
        return self.web_search.format_results(
            result['results'], 
            format_type='markdown'
        )
    
    def gather_knowledge(self, query: str, visual_analysis: str) -> dict:
        """
        Combina conhecimento de múltiplas fontes.
        
        Esta é a função principal chamada pelo Orchestrator.
        OTIMIZADO: Skip RAG se não há documentos + apenas 1 web search.
        """
        import time
        t_start = time.perf_counter()
        
        # Pesquisa enriquecida com contexto visual
        enriched_query = f"{query}. Observações: {visual_analysis[:200]}"
        
        doc_results = []
        doc_context = ""
        search_results = []
        
        # ═══════════════════════════════════════════════════════
        # 1. RAG LOCAL - SKIP se não há documentos
        # ═══════════════════════════════════════════════════════
        t1 = time.perf_counter()
        rag_stats = self.rag_tool.get_stats()
        if rag_stats.get('total_documents', 0) > 0:
            doc_results = self.search_documents(enriched_query, k=3)
            doc_context = self.rag_tool.get_relevant_context(enriched_query, max_tokens=1000)
            print(f"      [Knowledge] RAG local: {time.perf_counter()-t1:.2f}s ({len(doc_results)} docs)")
        else:
            print(f"      [Knowledge] RAG local: SKIP (0 documentos)")
        
        # ═══════════════════════════════════════════════════════
        # 2. WEB SEARCH - UMA ÚNICA pesquisa (otimizado)
        # ═══════════════════════════════════════════════════════
        t2 = time.perf_counter()
        web_results = ""
        web_context = ""
        try:
            # Fazer apenas UMA pesquisa
            search_results = self.web_search.search(query, max_results=5)
            if search_results:
                web_context = self.web_search.format_results(search_results, format_type='plain')
                web_results = web_context
                print(f"      [Knowledge] Web search: {time.perf_counter()-t2:.2f}s ({len(search_results)} resultados)")
                # Debug: mostrar títulos encontrados
                for i, r in enumerate(search_results[:3], 1):
                    print(f"         {i}. {r.title[:60]}...")
            else:
                print(f"      [Knowledge] Web search: {time.perf_counter()-t2:.2f}s (0 resultados)")
        except Exception as e:
            print(f"      ⚠️ Web search falhou: {e}")
        
        print(f"      [Knowledge] TOTAL: {time.perf_counter()-t_start:.2f}s")
        
        return {
            "local_documents": doc_results,
            "local_context": doc_context,
            "web_search": web_results,
            "web_context": web_context,
            "query_used": enriched_query,
            "sources_count": {
                "documents": len(doc_results),
                "web": len(search_results) if 'search_results' in dir() else 0
            }
        }
    
    def add_case_to_knowledge(self, case_summary: str, diagnosis: str):
        """
        Adiciona um caso resolvido à base de conhecimento.
        Útil para aprendizagem contínua.
        """
        content = f"""
## Caso Clínico

### Resumo
{case_summary}

### Diagnóstico
{diagnosis}
"""
        self.rag_tool.add_document(
            content=content,
            metadata={"type": "case_study"},
            source="casos_clinicos"
        )
        print("✅ Caso adicionado à base de conhecimento")
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do sistema de conhecimento"""
        return {
            "rag": self.rag_tool.get_stats(),
            "web_providers": self.web_search.get_available_providers()
        }