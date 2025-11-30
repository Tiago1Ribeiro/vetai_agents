"""
Agente de Visão - Analisa imagens veterinárias.
Versão otimizada para performance.
"""

import time
from pathlib import Path
from PIL import Image
from config.settings import settings


class VisionAgent:
    """Agente especializado em análise de imagens veterinárias"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
    
    def __init__(self):
        pass  # Não precisa de inicialização pesada
    
    def _optimize_image(self, image_path: str, max_size: int = 800) -> Image.Image:
        """Otimiza imagem para upload rápido"""
        img = Image.open(image_path)
        
        # Converter para RGB se necessário
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Redimensionar se muito grande
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
        
        return img
    
    def analyze_image(
        self, 
        image_paths: list[str], 
        animal_info: dict,
        symptoms: str,
        model: str = None  # Modelo opcional
    ) -> dict:
        """Analisa imagens do animal e extrai observações clínicas."""
        
        t_start = time.perf_counter()
        
        # Validar e carregar imagens
        valid_images = []
        for path in image_paths:
            try:
                print(f"   📂 Path recebido: {path}")
                p = Path(path)
                if p.exists() and p.suffix.lower() in self.SUPPORTED_FORMATS:
                    t_load = time.perf_counter()
                    img = self._optimize_image(path)
                    valid_images.append(img)
                    print(f"   📷 Imagem carregada: {p.name} ({img.size}) em {time.perf_counter()-t_load:.2f}s")
                else:
                    print(f"   ⚠️ Path inválido ou formato não suportado: {path}")
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar {path}: {e}")
        
        if not valid_images:
            return {
                "visual_analysis": "Nenhuma imagem válida fornecida.",
                "images_analyzed": 0
            }
        
        t_loaded = time.perf_counter()
        print(f"   ⏱️ Imagens carregadas em {t_loaded - t_start:.2f}s")
        
        # Criar prompt
        prompt = f"""És um veterinário experiente a analisar imagens clínicas.

**Informação do Animal:**
- Espécie: {animal_info.get('especie', 'Não especificado')}
- Raça: {animal_info.get('raca', 'Não especificado')}
- Idade: {animal_info.get('idade', 'Não especificado')}
- Peso: {animal_info.get('peso', 'Não especificado')}

**Sintomas Reportados:** {symptoms}

Analisa as imagens e descreve:
1. Observações visuais objetivas
2. Localização das alterações
3. Gravidade aparente (Leve/Moderada/Grave/Urgente)

Sê conciso e objetivo."""

        # Chamar API apropriada
        t_api = time.perf_counter()
        # Usar modelo passado ou default
        vision_model = model or settings.VLM_MODEL
        
        # Detectar provider pelo nome do modelo
        if vision_model.startswith("gemini"):
            response = self._call_gemini(prompt, valid_images, vision_model)
            api_name = "Gemini"
        elif vision_model.startswith("pixtral"):
            response = self._call_mistral_vision(prompt, valid_images, vision_model)
            api_name = "Mistral"
        else:
            # Fallback: OpenRouter
            response = self._call_openrouter_vision(prompt, valid_images, vision_model)
            api_name = "OpenRouter"
        
        t_done = time.perf_counter()
        
        print(f"   ⏱️ API {api_name}: {t_done - t_api:.2f}s")
        print(f"   ⏱️ Total visão: {t_done - t_start:.2f}s")
        
        return {
            "visual_analysis": response,
            "images_analyzed": len(valid_images),
            "model_used": vision_model
        }
    
    def _call_gemini(self, prompt: str, images: list[Image.Image], model: str = None) -> str:
        """Chama Gemini Vision API via HTTP direto (mais rápido)"""
        import base64
        import io
        import httpx
        
        t0 = time.perf_counter()
        
        # Usar modelo passado ou default
        vision_model = model or settings.VLM_MODEL
        
        # Converter imagens para base64
        image_parts = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)
            b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            image_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64
                }
            })
        print(f"      [Gemini] Images to base64: {time.perf_counter()-t0:.2f}s")
        
        # Construir request - usar modelo selecionado
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{vision_model}:generateContent?key={settings.GOOGLE_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}] + image_parts
            }],
            "generationConfig": {
                "maxOutputTokens": 2048,
                "temperature": 0.3
            }
        }
        
        t1 = time.perf_counter()
        print(f"      [Gemini] Sending HTTP request...")
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                
            result = response.json()
            
            # Debug: mostrar estrutura da resposta
            print(f"      [Gemini] Response keys: {result.keys()}")
            
            # Verificar se há erro na resposta
            if "error" in result:
                error_msg = result["error"].get("message", "Erro desconhecido")
                print(f"      [Gemini] API Error: {error_msg}")
                return f"Erro da API Gemini: {error_msg}"
            
            # Extrair texto da resposta
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        text = parts[0]["text"]
                        print(f"      [Gemini] Response received: {time.perf_counter()-t1:.2f}s")
                        return text
                
                # Verificar se foi bloqueado
                if "finishReason" in candidate:
                    reason = candidate["finishReason"]
                    if reason == "SAFETY":
                        return "A imagem foi bloqueada pelos filtros de segurança do Gemini."
                    return f"Resposta incompleta. Razão: {reason}"
            
            # Se chegou aqui, estrutura inesperada
            print(f"      [Gemini] Unexpected response structure: {result}")
            return "Não foi possível processar a resposta da API."
            
        except httpx.TimeoutException:
            return "Erro: Timeout na chamada ao Gemini (>30s)"
        except httpx.HTTPStatusError as e:
            print(f"      [Gemini] HTTP Error: {e.response.status_code} - {e.response.text}")
            return f"Erro HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            print(f"      [Gemini] Exception: {type(e).__name__}: {e}")
            return f"Erro na análise: {type(e).__name__}: {str(e)}"

    def _call_mistral_vision(self, prompt: str, images: list, model: str) -> str:
        """Chama Mistral Vision API (Pixtral)"""
        import base64
        import io
        import httpx
        
        t0 = time.perf_counter()
        
        # Converter imagens para base64 com formato URL data
        image_content = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)
            b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            image_content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{b64}"
            })
        
        print(f"      [Mistral] Images to base64: {time.perf_counter()-t0:.2f}s")
        
        # Construir mensagem com imagens + texto
        messages = [{
            "role": "user",
            "content": image_content + [{"type": "text", "text": prompt}]
        }]
        
        t1 = time.perf_counter()
        print(f"      [Mistral] Sending request to {model}...")
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": 1500,
                        "temperature": 0.2
                    }
                )
                response.raise_for_status()
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            print(f"      [Mistral] Response received: {time.perf_counter()-t1:.2f}s")
            return text
            
        except httpx.TimeoutException:
            return "Erro: Timeout na chamada ao Mistral (>60s)"
        except Exception as e:
            return f"Erro na análise: {str(e)}"

    def _call_openrouter_vision(self, prompt: str, images: list, model: str) -> str:
        """Chama OpenRouter Vision API para modelos como Qwen"""
        import base64
        import io
        import httpx
        
        t0 = time.perf_counter()
        
        # Converter imagens para base64 com formato URL data
        image_content = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)
            b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })
        
        print(f"      [OpenRouter] Images to base64: {time.perf_counter()-t0:.2f}s")
        
        # Construir mensagem com imagens + texto
        messages = [{
            "role": "user",
            "content": image_content + [{"type": "text", "text": prompt}]
        }]
        
        t1 = time.perf_counter()
        print(f"      [OpenRouter] Sending request to {model}...")
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "VetVision"
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": 1500,
                        "temperature": 0.2
                    }
                )
                response.raise_for_status()
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            print(f"      [OpenRouter] Response received: {time.perf_counter()-t1:.2f}s")
            return text
            
        except httpx.TimeoutException:
            return "Erro: Timeout na chamada ao OpenRouter (>60s)"
        except Exception as e:
            return f"Erro na análise: {str(e)}"
