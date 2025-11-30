"""
Ferramenta de pesquisa web para conhecimento veterinário.

Funcionalidades:
- Pesquisa web com DuckDuckGo (gratuito)
- Pesquisa com Gemini Grounding (Google AI)
- Formatação de resultados para LLM
"""

import httpx
from dataclasses import dataclass
from typing import Optional
from config.settings import settings


@dataclass
class WebSearchResult:
    """Resultado de uma pesquisa web"""
    title: str
    url: str
    snippet: str
    source: str


class WebSearchTool:
    """
    Ferramenta de pesquisa web para temas veterinários.
    
    Usa DuckDuckGo como fallback gratuito e Gemini para análise.
    """
    
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        preferred_provider: str = 'duckduckgo'
    ):
        """
        Args:
            google_api_key: API key do Google AI (opcional)
            preferred_provider: 'duckduckgo' ou 'gemini'
        """
        self.google_api_key = google_api_key or settings.GOOGLE_API_KEY
        self.preferred_provider = preferred_provider
        self.client = httpx.Client(timeout=30.0)
        
        # Cache de resultados
        self._cache = {}
    
    def search(
        self,
        query: str,
        max_results: int = 5
    ) -> list[WebSearchResult]:
        """
        Pesquisa web genérica.
        
        Args:
            query: Texto da pesquisa
            max_results: Número máximo de resultados
            
        Returns:
            Lista de WebSearchResult
        """
        # Verificar cache
        cache_key = f"{query}:{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            results = self._search_duckduckgo(query, max_results)
        except Exception as e:
            print(f"⚠️ DuckDuckGo falhou: {e}")
            results = []
        
        self._cache[cache_key] = results
        return results
    
    def _search_duckduckgo(
        self,
        query: str,
        max_results: int = 5
    ) -> list[WebSearchResult]:
        """Pesquisa usando DuckDuckGo (gratuito)"""
        import warnings
        warnings.filterwarnings("ignore", message=".*duckduckgo.*")
        warnings.filterwarnings("ignore", message=".*ddgs.*")
        
        try:
            from duckduckgo_search import DDGS
            
            # Enriquecer query com termos veterinários se não tiver
            vet_terms = ['veterinary', 'veterinário', 'vet', 'animal', 'dog', 'cat', 'diagnosis', 'treatment']
            has_vet_term = any(term.lower() in query.lower() for term in vet_terms)
            if not has_vet_term:
                query = f"veterinary {query} symptoms treatment"
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            return [
                WebSearchResult(
                    title=r.get('title', ''),
                    url=r.get('href', r.get('link', '')),
                    snippet=r.get('body', r.get('snippet', '')),
                    source='duckduckgo'
                )
                for r in results
            ]
        except ImportError as ie:
            print(f"⚠️ duckduckgo_search não instalado: {ie}")
            return []
        except Exception as e:
            print(f"⚠️ Erro DuckDuckGo: {e}")
            return []
    
    def search_veterinary(
        self,
        query: str,
        max_results: int = 5,
        use_gemini_analysis: bool = True
    ) -> dict:
        """
        Pesquisa especializada em temas veterinários.
        
        Args:
            query: Texto da pesquisa
            max_results: Número máximo de resultados
            use_gemini_analysis: Usar Gemini para analisar resultados
            
        Returns:
            Dict com resultados e análise opcional
        """
        # Enriquecer query com contexto veterinário específico
        # Adiciona termos para obter resultados mais relevantes
        vet_query = f"{query} site:vetmed.edu OR site:avma.org OR site:vin.com OR veterinary medicine symptoms treatment"
        
        results = self.search(vet_query, max_results)
        
        # Se não houver resultados, tentar query mais simples
        if not results:
            vet_query = f"veterinary {query}"
            results = self.search(vet_query, max_results)
        
        analysis = None
        if use_gemini_analysis and results and self.google_api_key:
            try:
                analysis = self._analyze_with_gemini(query, results)
            except Exception as e:
                print(f"⚠️ Análise Gemini falhou: {e}")
        
        return {
            'query': query,
            'results': results,
            'analysis': analysis
        }
    
    def _analyze_with_gemini(
        self,
        query: str,
        results: list[WebSearchResult]
    ) -> str:
        """Usa Gemini para analisar e sumarizar resultados"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.google_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            # Formatar resultados para o prompt
            results_text = "\n\n".join([
                f"**{r.title}**\n{r.snippet}\nFonte: {r.url}"
                for r in results
            ])
            
            prompt = f"""Analisa os seguintes resultados de pesquisa sobre "{query}" e fornece um resumo conciso das informações mais relevantes para um veterinário.

{results_text}

Fornece um resumo estruturado com:
1. Informações principais
2. Dados relevantes para diagnóstico
3. Fontes mais confiáveis"""

            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Erro na análise Gemini: {e}")
    
    def format_results(
        self,
        results: list[WebSearchResult],
        format_type: str = 'markdown'
    ) -> str:
        """
        Formata resultados para exibição.
        
        Args:
            results: Lista de resultados
            format_type: 'markdown', 'plain', 'html'
        """
        if not results:
            return "Nenhum resultado encontrado."
        
        if format_type == 'markdown':
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"### {i}. {r.title}")
                lines.append(f"{r.snippet}")
                lines.append(f"*Fonte: [{r.url}]({r.url})*\n")
            return "\n".join(lines)
        
        elif format_type == 'plain':
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r.title}")
                lines.append(f"   {r.snippet}")
                lines.append(f"   Fonte: {r.url}\n")
            return "\n".join(lines)
        
        else:
            return str(results)
    
    def get_context_for_llm(
        self,
        query: str,
        max_chars: int = 2000
    ) -> str:
        """
        Obtém contexto formatado para LLM.
        
        Args:
            query: Texto da pesquisa
            max_chars: Limite de caracteres
            
        Returns:
            Texto formatado
        """
        results = self.search(query, max_results=5)
        
        context_parts = []
        current_length = 0
        
        for r in results:
            snippet = f"[{r.title}]: {r.snippet}"
            if current_length + len(snippet) > max_chars:
                break
            context_parts.append(snippet)
            current_length += len(snippet)
        
        return "\n\n".join(context_parts)
    
    def get_available_providers(self) -> list[str]:
        """Retorna lista de providers disponíveis"""
        providers = ['duckduckgo']
        if self.google_api_key:
            providers.append('gemini')
        return providers


# Função utilitária
def quick_search(query: str) -> list[WebSearchResult]:
    """Pesquisa rápida com configuração padrão"""
    tool = WebSearchTool()
    return tool.search(query)
