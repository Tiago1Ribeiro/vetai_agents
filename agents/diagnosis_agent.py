import httpx
from config.settings import settings

class DiagnosisAgent:
    """Agente de raciocínio clínico veterinário"""
    
    def __init__(self):
        self.client = httpx.Client(timeout=60.0)  # Timeout de 60s por chamada
    
    def generate_diagnosis(
        self,
        animal_info: dict,
        symptoms: str,
        visual_analysis: str,
        knowledge: dict,
        model: str = None  # Modelo opcional
    ) -> dict:
        """
        Gera diagnóstico diferencial baseado em todas as informações
        """
        
        # Formatar conhecimento
        doc_context = "\n".join([
            f"[Documento: {d['source']}]\n{d['content']}"
            for d in knowledge.get("local_documents", [])[:3]
        ])
        
        web_context = knowledge.get("web_search", "")
        
        prompt = f"""És um veterinário experiente a realizar um diagnóstico diferencial.

## INFORMAÇÃO DO CASO

### Dados do Animal:
- Espécie: {animal_info.get('especie')}
- Raça: {animal_info.get('raca')}
- Idade: {animal_info.get('idade')}
- Peso: {animal_info.get('peso')}
- Histórico médico: {animal_info.get('historico', 'Não disponível')}

### Sintomas Reportados pelo Tutor:
{symptoms}

### Análise Visual das Imagens:
{visual_analysis}

### Informação de Referência (Literatura e Web):
{doc_context}

{web_context}

---

## TAREFA

Com base em toda a informação, fornece:

### 1. DIAGNÓSTICOS DIFERENCIAIS
Lista os 3-5 diagnósticos mais prováveis, ordenados por probabilidade:
- Para cada um: nome, probabilidade estimada (%), justificação

### 2. EXAMES RECOMENDADOS
Que exames/testes confirmariam o diagnóstico:
- Análises laboratoriais
- Imagiologia
- Outros testes

### 3. TRATAMENTO INICIAL
Sugestões de tratamento/manejo enquanto não há diagnóstico definitivo:
- Cuidados imediatos
- Medicação sintomática (se aplicável)
- O que NÃO fazer

### 4. NÍVEL DE URGÊNCIA
Classifica: 🟢 Rotina | 🟡 Consulta em 24-48h | 🔴 Urgente | ⚫ Emergência

### 5. PRÓXIMOS PASSOS
Recomendações claras para o tutor

### 6. DISCLAIMER
Lembra que isto é uma orientação e não substitui consulta presencial.

---
Raciocina passo a passo antes de concluir."""

        # Tentar múltiplos providers em sequência
        model_used = "unknown"
        response = None
        
        # Se um modelo específico foi passado, usar diretamente
        if model:
            try:
                model_short = model.split("/")[-1].split(":")[0]
                print(f"   Usando modelo selecionado: {model_short}...")
                
                # Determinar provider pelo formato do modelo
                if model.startswith("mistral"):
                    response = self._call_mistral(prompt, model)
                    model_used = model_short
                elif model.startswith("gemini"):
                    response = self._call_gemini(prompt)
                    model_used = model_short
                else:
                    # Assumir OpenRouter
                    response = self._call_openrouter(prompt, model)
                    model_used = model_short
                    
            except Exception as e:
                print(f"   Modelo selecionado falhou: {e}")
                # Continuar com fallback
        
        # Fallback: tentar múltiplos providers
        if response is None:
            # Lista de modelos OpenRouter para tentar (gratuitos)
            openrouter_models = [
                (settings.LLM_OPENROUTER_1, "grok-4.1-fast"),
                (settings.LLM_OPENROUTER_3, "gemma-3-27b"),
                (settings.LLM_OPENROUTER_2, "deepseek-r1-chimera"),
                (settings.LLM_OPENROUTER_4, "glm-4.5-air"),
            ]
            
            # 1. Tentar modelos OpenRouter
            for llm_model, name in openrouter_models:
                try:
                    print(f"   Tentando {name}...")
                    response = self._call_openrouter(prompt, llm_model)
                    model_used = name
                    break
                except Exception as e:
                    print(f"   {name} falhou: {e}")
                    continue
        
        # 2. Se OpenRouter falhou, tentar Mistral
        if response is None:
            try:
                print("   Tentando Mistral...")
                response = self._call_mistral(prompt)
                model_used = "mistral-small"
            except Exception as e2:
                print(f"   Mistral falhou: {e2}")
                
                # 3. Tentar Gemini
                try:
                    print("   Tentando Gemini...")
                    response = self._call_gemini(prompt)
                    model_used = "gemini"
                except Exception as e3:
                    print(f"   Gemini falhou: {e3}")
                    response = "❌ Não foi possível gerar diagnóstico. Verifique as API keys."
                    model_used = "none"
        
        return {
            "diagnosis_report": response,
            "model_used": model_used
        }
    
    def _call_openrouter(self, prompt: str, model: str = None) -> str:
        """Chama modelo via OpenRouter (gratuito)"""
        model = model or settings.LLM_OPENROUTER_1  # tngtech/deepseek-r1t-chimera:free
        response = self.client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "VetDiagnosis"
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "És um veterinário especialista em diagnóstico."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4000,
                "temperature": 0.2
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_mistral(self, prompt: str, model: str = None) -> str:
        """Backup: Mistral AI"""
        mistral_model = model or settings.LLM_BACKUP  # mistral-small-latest
        response = self.client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": mistral_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "És um veterinário especialista em diagnóstico."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4000,
                "temperature": 0.2
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_gemini(self, prompt: str) -> str:
        """Backup: Gemini"""
        import google.generativeai as genai
        
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        model = genai.GenerativeModel(settings.VLM_MODEL) 
        
        response = model.generate_content(prompt)
        return response.text