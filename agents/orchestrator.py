from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
import hashlib
import time

from agents.vision_agent import VisionAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.diagnosis_agent import DiagnosisAgent


# ============================================================
# CONFIGURAÇÃO BASEADA NA URGÊNCIA
# ============================================================

@dataclass 
class DiagnosisConfig:
    """Configuração do diagnóstico baseada na urgência"""
    max_web_results: int = 5
    vision_timeout: int = 30
    knowledge_timeout: int = 20
    diagnosis_timeout: int = 60
    detailed_research: bool = True
    
    @classmethod
    def for_urgency(cls, urgency: str) -> "DiagnosisConfig":
        """Retorna configuração otimizada para o nível de urgência"""
        configs = {
            "🔴 Urgente": cls(
                max_web_results=3,
                vision_timeout=15,
                knowledge_timeout=10,
                diagnosis_timeout=30,
                detailed_research=False  # Mais rápido
            ),
            "🟡 Moderada": cls(
                max_web_results=5,
                vision_timeout=25,
                knowledge_timeout=15,
                diagnosis_timeout=45,
                detailed_research=True
            ),
            "🟢 Rotina": cls(
                max_web_results=8,
                vision_timeout=30,
                knowledge_timeout=25,
                diagnosis_timeout=60,
                detailed_research=True
            )
        }
        return configs.get(urgency, configs["🟢 Rotina"])


# ============================================================
# QUERY BUILDER VETERINÁRIO
# ============================================================

class VetQueryBuilder:
    """Constrói queries otimizadas para pesquisa veterinária"""
    
    # Termos médicos por categoria de sintoma (PT -> EN médico)
    MEDICAL_TERMS = {
        "vomito": ["emesis", "vomiting", "gastric"],
        "vómito": ["emesis", "vomiting", "gastric"],
        "diarreia": ["diarrhea", "enteritis", "colitis"],
        "tosse": ["cough", "respiratory", "bronchitis"],
        "coceira": ["pruritus", "dermatitis", "itching"],
        "coçar": ["pruritus", "dermatitis", "scratching"],
        "claudicação": ["lameness", "orthopedic", "limping"],
        "mancar": ["lameness", "limping"],
        "letargia": ["lethargy", "weakness", "malaise"],
        "febre": ["fever", "pyrexia", "infection"],
        "perda de apetite": ["anorexia", "inappetence"],
        "não come": ["anorexia", "inappetence"],
        "perda de peso": ["weight loss", "cachexia"],
        "convulsões": ["seizures", "epilepsy", "neurological"],
        "tremores": ["tremors", "shaking", "neurological"],
        "lesão": ["lesion", "wound", "skin"],
        "ferida": ["wound", "laceration", "injury"],
        "inchaço": ["swelling", "edema", "inflammation"],
        "sangue": ["bleeding", "hemorrhage", "hematuria"],
        "urina": ["urinary", "dysuria", "UTI"],
        "olhos": ["ocular", "conjunctivitis", "eye"],
        "ouvido": ["otitis", "ear", "auricular"],
    }
    
    SPECIES_MAP = {
        "Cão": "canine dog",
        "Gato": "feline cat",
        "Outro": "veterinary animal",
    }
    
    @classmethod
    def build_query(cls, case, focus: str = "diagnosis") -> str:
        """Constrói query otimizada para pesquisa veterinária"""
        parts = []
        
        # 1. Espécie
        species = cls.SPECIES_MAP.get(case.especie, "veterinary")
        parts.append(species.split()[0])  # canine, feline, etc.
        
        # 2. Sintomas - converter para termos médicos ingleses
        sintomas_lower = case.sintomas.lower()
        medical_found = []
        
        for pt_term, en_terms in cls.MEDICAL_TERMS.items():
            if pt_term in sintomas_lower:
                medical_found.extend(en_terms[:2])  # Max 2 termos por sintoma
        
        if medical_found:
            # Usar termos médicos encontrados
            parts.extend(list(set(medical_found))[:4])  # Deduplica, max 4
        else:
            # Fallback: usar "veterinary" + sintomas simplificados
            parts.append("veterinary")
            words = [w for w in case.sintomas.split()[:4] if len(w) > 3]
            parts.extend(words)
        
        # 3. Foco da pesquisa
        focus_terms = {
            "diagnosis": "differential diagnosis",
            "treatment": "treatment therapy",
            "emergency": "emergency urgent critical"
        }
        parts.append(focus_terms.get(focus, "diagnosis"))
        
        # 4. Idade se relevante
        idade_lower = case.idade.lower()
        if any(t in idade_lower for t in ["filhote", "puppy", "kitten", "meses", "semanas"]):
            parts.append("puppy" if case.especie == "Cão" else "kitten")
        elif any(t in idade_lower for t in ["senior", "idoso", "velho", "12 anos", "13 anos", "14 anos", "15 anos"]):
            parts.append("geriatric senior")
        
        return " ".join(parts)


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class CaseInput:
    """Input para um caso veterinário"""
    especie: str
    raca: str = "Desconhecida"
    idade: str = "Desconhecida"
    peso: str = "Desconhecido"
    sexo: str = "Desconhecido"
    castrado: bool = False
    historico: str = ""
    sintomas: str = ""
    urgencia: str = "🟢 Rotina"
    image_paths: List[str] = field(default_factory=list)
    tipos_imagem: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "especie": self.especie,
            "raca": self.raca,
            "idade": self.idade,
            "peso": self.peso,
            "sexo": self.sexo,
            "castrado": self.castrado,
            "historico": self.historico,
            "urgencia": self.urgencia
        }
    
    def get_cache_key(self) -> str:
        """Gera chave única para cache"""
        content = f"{self.especie}:{self.sintomas}:{self.historico}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

class VetDiagnosisOrchestrator:
    """
    Orquestrador principal do sistema de diagnóstico
    Coordena os agentes e mantém o estado do caso
    
    Melhorias v2:
    - Configuração baseada na urgência
    - Query builder veterinário otimizado
    - Logging com timestamps
    - Fallback diagnosis
    """
    
    def __init__(self):
        self.vision_agent = VisionAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.diagnosis_agent = DiagnosisAgent()
        
        self.case_history = []
    
    def _log(self, emoji: str, message: str):
        """Logging formatado com timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {emoji} {message}")
    
    def run_diagnosis(self, case: CaseInput, vision_model: str = None, text_model: str = None) -> dict:
        """
        Executa o pipeline completo de diagnóstico
        
        Args:
            case: Dados do caso clínico
            vision_model: Modelo para análise de imagens (opcional)
            text_model: Modelo para diagnóstico/texto (opcional)
        """
        start_time = time.time()
        
        # Configuração baseada na urgência
        config = DiagnosisConfig.for_urgency(case.urgencia)
        
        self._log("🏥", f"Iniciando diagnóstico [{case.urgencia}]")
        self._log("🐾", f"{case.especie} - {case.raca} - {case.idade}")
        if vision_model:
            self._log("👁️", f"Modelo Visão: {vision_model}")
        if text_model:
            self._log("🧠", f"Modelo Texto: {text_model}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "case_info": case.to_dict(),
            "symptoms": case.sintomas,
            "urgency": case.urgencia,
            "steps": [],
            "performance": {}
        }
        
        # ════════════════════════════════════════════════════
        # PASSO 1: Análise de Imagem
        # ════════════════════════════════════════════════════
        self._log("📸", "Passo 1: Analisando imagens...")
        t1 = time.time()
        
        if case.image_paths:
            try:
                visual_result = self.vision_agent.analyze_image(
                    image_paths=case.image_paths,
                    animal_info=case.to_dict(),
                    symptoms=case.sintomas,
                    model=vision_model  # Passar modelo selecionado
                )
                results["visual_analysis"] = visual_result["visual_analysis"]
                results["steps"].append({
                    "step": "vision_analysis",
                    "status": "success",
                    "images_analyzed": visual_result["images_analyzed"],
                    "model_used": vision_model or "default",
                    "duration_ms": int((time.time() - t1) * 1000)
                })
                self._log("✅", f"Analisadas {visual_result['images_analyzed']} imagens ({int((time.time()-t1)*1000)}ms)")
            except Exception as e:
                self._log("⚠️", f"Erro na análise visual: {str(e)[:50]}")
                results["visual_analysis"] = "Erro na análise de imagem"
                results["steps"].append({
                    "step": "vision_analysis",
                    "status": "failed",
                    "error": str(e)
                })
        else:
            results["visual_analysis"] = "Nenhuma imagem fornecida"
            results["steps"].append({
                "step": "vision_analysis",
                "status": "skipped"
            })
            self._log("⚠️", "Nenhuma imagem fornecida")
        
        # ════════════════════════════════════════════════════
        # PASSO 2: Recolha de Conhecimento
        # ════════════════════════════════════════════════════
        self._log("📚", "Passo 2: Pesquisando informação...")
        t2 = time.time()
        
        # Query otimizada com VetQueryBuilder
        focus = "emergency" if "Urgente" in case.urgencia else "diagnosis"
        query = VetQueryBuilder.build_query(case, focus=focus)
        self._log("🔍", f"Query: {query[:60]}...")
        
        try:
            knowledge = self.knowledge_agent.gather_knowledge(
                query=query,
                visual_analysis=results["visual_analysis"]
            )
            
            web_found = len(knowledge.get("web_search", "")) > 0
            results["knowledge_gathered"] = {
                "documents_found": len(knowledge["local_documents"]),
                "web_search_done": web_found,
                "web_results_chars": len(knowledge.get("web_search", ""))
            }
            
            # Guardar contexto web para o relatório
            if web_found:
                results["research"] = knowledge.get("web_search", "")
            
            results["steps"].append({
                "step": "knowledge_gathering",
                "status": "success",
                "duration_ms": int((time.time() - t2) * 1000)
            })
            
            self._log("✅", f"Docs locais: {len(knowledge['local_documents'])} | Web: {len(knowledge.get('web_search', ''))} chars ({int((time.time()-t2)*1000)}ms)")
            
        except Exception as e:
            self._log("⚠️", f"Erro na pesquisa: {str(e)[:50]}")
            knowledge = {"local_documents": [], "web_search": ""}
            results["steps"].append({
                "step": "knowledge_gathering",
                "status": "failed",
                "error": str(e)
            })
        
        # ════════════════════════════════════════════════════
        # PASSO 3: Diagnóstico
        # ════════════════════════════════════════════════════
        self._log("🩺", "Passo 3: Gerando diagnóstico...")
        t3 = time.time()
        
        try:
            diagnosis = self.diagnosis_agent.generate_diagnosis(
                animal_info=case.to_dict(),
                symptoms=case.sintomas,
                visual_analysis=results["visual_analysis"],
                knowledge=knowledge,
                model=text_model  # Passar modelo selecionado
            )
            results["diagnosis"] = diagnosis["diagnosis_report"]
            results["steps"].append({
                "step": "diagnosis",
                "status": "success",
                "model": diagnosis["model_used"],
                "duration_ms": int((time.time() - t3) * 1000)
            })
            self._log("✅", f"Diagnóstico gerado com {diagnosis['model_used']} ({int((time.time()-t3)*1000)}ms)")
            
        except Exception as e:
            self._log("⚠️", f"Erro no diagnóstico: {str(e)[:50]}")
            # Usar fallback
            results["diagnosis"] = self._generate_fallback_diagnosis(case, results)
            results["steps"].append({
                "step": "diagnosis",
                "status": "fallback",
                "error": str(e)
            })
        
        # ════════════════════════════════════════════════════
        # FINALIZAÇÃO
        # ════════════════════════════════════════════════════
        total_time = int((time.time() - start_time) * 1000)
        results["performance"] = {
            "total_ms": total_time,
            "total_seconds": round(total_time / 1000, 1)
        }
        
        # Guardar no histórico
        self.case_history.append(results)
        
        self._log("🏁", f"Diagnóstico completo em {total_time}ms ({total_time/1000:.1f}s)")
        
        return results
    
    def _generate_fallback_diagnosis(self, case: CaseInput, partial_results: dict) -> str:
        """Gera um diagnóstico básico quando o principal falha"""
        
        urgency_msg = ""
        if "Urgente" in case.urgencia:
            urgency_msg = """
⚠️ **CASO MARCADO COMO URGENTE**
Recomenda-se procurar atendimento veterinário imediato.
"""
        
        return f"""
## ⚠️ Diagnóstico de Contingência

Devido a limitações técnicas temporárias, não foi possível gerar um diagnóstico detalhado.

### Dados do Caso
- **Animal:** {case.especie} ({case.raca})
- **Idade:** {case.idade} | **Peso:** {case.peso}

### Sintomas Reportados
{case.sintomas}

{urgency_msg}

### Recomendações Gerais

1. **Consulte um médico veterinário presencialmente**
2. Mantenha o animal em observação
3. Registe qualquer alteração nos sintomas
4. Garanta hidratação e conforto
5. Não administre medicação sem orientação profissional

---
*Este é um diagnóstico de contingência. Consulte sempre um profissional veterinário.*
"""
    
    def print_report(self, results: dict):
        """Imprime o relatório formatado"""
        print("=" * 70)
        print("            RELATÓRIO DE DIAGNÓSTICO VETERINÁRIO")
        print("=" * 70)
        print(f"\n📅 Data: {results['timestamp']}")
        print(f"🐾 Animal: {results['case_info']['especie']} - {results['case_info']['raca']}")
        print(f"📊 Idade: {results['case_info']['idade']} | Peso: {results['case_info']['peso']}")
        
        print("\n" + "-" * 70)
        print("📝 SINTOMAS REPORTADOS:")
        print("-" * 70)
        print(results['symptoms'])
        
        print("\n" + "-" * 70)
        print("👁️ ANÁLISE VISUAL:")
        print("-" * 70)
        print(results['visual_analysis'])
        
        print("\n" + "-" * 70)
        print("🩺 DIAGNÓSTICO E RECOMENDAÇÕES:")
        print("-" * 70)
        print(results['diagnosis'])
        
        print("\n" + "=" * 70)
        print("⚠️  AVISO: Esta análise é apenas orientativa.")
        print("    Consulte sempre um veterinário presencialmente.")
        print("=" * 70)