# ui.py - Interface Gradio Clean & Profissional

# Suprimir warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*duckduckgo.*")
warnings.filterwarnings("ignore", message=".*LangChain.*")

import gradio as gr
from datetime import datetime
from typing import Optional
import json

from agents.orchestrator import VetDiagnosisOrchestrator, CaseInput
from config.settings import settings

# Inicializar sistema
vet_system = VetDiagnosisOrchestrator()

# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Modelos disponíveis para Visão (VLM)
VISION_MODELS = {
    "Gemini Flash (Google)": "gemini-2.5-flash",
    "Gemini Flash Lite (Google)": "gemini-2.5-flash-lite",
    "Pixtral 12B (Mistral)": "pixtral-12b-2409",
}

# Modelos disponíveis para Texto/Diagnóstico (LLM)
TEXT_MODELS = {
    "Grok 4.1 Fast (xAI)": "x-ai/grok-4.1-fast:free",
    "Gemma 3 27B (Google)": "google/gemma-3-27b-it:free",
    "DeepSeek R1 Chimera": "tngtech/deepseek-r1t-chimera:free",
    "GLM 4.5 Air": "z-ai/glm-4.5-air:free",
    "Mistral Small": "mistral-small-latest",
}

# ============================================================
# CONFIGURAÇÕES
# ============================================================

ESPECIES = {
    "Cão": {
        "icon": "🐕",
        "racas": ["SRD (Sem Raça Definida)", "Labrador Retriever", "Pastor Alemão", 
                  "Golden Retriever", "Bulldog Francês", "Poodle", "Beagle", 
                  "Rottweiler", "Yorkshire Terrier", "Boxer", "Husky Siberiano",
                  "Pitbull", "Border Collie", "Shih Tzu", "Dachshund", "Outra"]
    },
    "Gato": {
        "icon": "🐈",
        "racas": ["SRD (Sem Raça Definida)", "Persa", "Siamês", "Maine Coon", 
                  "Ragdoll", "British Shorthair", "Bengal", "Sphynx", 
                  "Scottish Fold", "Abissínio", "Outra"]
    },
    "Ave": {
        "icon": "🦜",
        "racas": ["Periquito", "Canário", "Calopsita", "Papagaio", "Outra"]
    },
    "Outro": {
        "icon": "🐾",
        "racas": ["Especificar no histórico"]
    }
}

SINTOMAS = {
    "Gerais": ["Letargia", "Perda de apetite", "Perda de peso", "Febre"],
    "Digestivos": ["Vómitos", "Diarreia", "Sangue nas fezes"],
    "Respiratórios": ["Tosse", "Espirros", "Dificuldade respiratória"],
    "Pele": ["Coceira", "Perda de pelo", "Lesões", "Vermelhidão"],
    "Movimento": ["Claudicação", "Rigidez", "Dor ao mover"]
}

# ============================================================
# CSS MODERNO (cores corrigidas para a11y)
# ============================================================

CSS = """
/* === VARIÁVEIS === */
:root {
    --primary: #0d9488;
    --primary-hover: #0f766e;
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-light: #475569;
    --success: #047857;
    --warning: #b45309;
    --error: #b91c1c;
}

/* === BASE === */
.gradio-container {
    background: var(--bg) !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
}

/* === HEADER (cor do texto corrigida) === */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
    color: #ffffff;
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.app-header h1 {
    margin: 0 0 0.25rem 0;
    font-size: 1.75rem;
    font-weight: 700;
    color: #ffffff !important;
}

.app-header p {
    margin: 0;
    font-size: 0.95rem;
    color: #e2e8f0 !important;
}


/* === DISCLAIMER (contraste corrigido) === */
.disclaimer {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 4px solid #d97706;
    padding: 1rem 1.25rem;
    border-radius: 0 12px 12px 0;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
    color: #78350f;
}

.disclaimer strong {
    color: #92400e;
}

/* === SECTIONS === */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* === RESULTS === */
.results-header {
    background: linear-gradient(135deg, var(--primary) 0%, #0f766e 100%);
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    font-weight: 600;
    font-size: 1.1rem;
}

/* === LOADING (contraste corrigido) === */
.loading-box {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}

.loading-box .spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #bfdbfe;
    border-top-color: #2563eb;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 1rem;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.loading-box .title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e3a8a;
    margin-bottom: 0.25rem;
}

.loading-box .subtitle {
    color: #1e40af;
    font-size: 0.9rem;
}

/* === SUMMARY === */
.case-summary {
    background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
    border: 1px solid #99f6e4;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}

.case-summary table {
    width: 100%;
    border-collapse: collapse;
}

.case-summary td {
    padding: 0.35rem 0.5rem;
    font-size: 0.9rem;
}

.case-summary td:first-child {
    font-weight: 600;
    color: var(--text);
    width: 40%;
}

.case-summary td:last-child {
    color: var(--text-light);
}

/* === FOOTER === */
.app-footer {
    text-align: center;
    padding: 1.5rem;
    color: var(--text-light);
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}

/* === BUTTONS === */
.primary-btn {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 1rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3) !important;
}

/* === FINAL DISCLAIMER (contraste corrigido) === */
.final-disclaimer {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
    font-size: 0.85rem;
    color: #7f1d1d;
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
}

.final-disclaimer .icon {
    font-size: 1.25rem;
}

/* === PERFORMANCE (contraste corrigido) === */
.perf-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #ecfdf5;
    color: #047857;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* === MODEL BADGE === */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #ede9fe;
    color: #5b21b6;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-left: 0.5rem;
}

/* === CONFIG SECTION === */
.config-section {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
}

.config-section .section-subtitle {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-light);
    margin-bottom: 0.5rem;
}

/* === FOCUS STATES (a11y) === */
*:focus-visible {
    outline: 2px solid var(--primary) !important;
    outline-offset: 2px !important;
}

/* === DROPDOWN FIX (Gradio 5.x) === */
.gradio-dropdown {
    font-size: 0.95rem !important;
}

.gradio-dropdown button[aria-haspopup="listbox"],
.gradio-dropdown .wrap {
    min-height: 42px !important;
    max-height: 42px !important;
}

.gradio-dropdown svg {
    width: 18px !important;
    height: 18px !important;
    max-width: 18px !important;
    max-height: 18px !important;
}

.gradio-dropdown .icon-wrap svg,
.gradio-dropdown button svg,
.gradio-dropdown .dropdown-arrow {
    width: 16px !important;
    height: 16px !important;
}

/* Fix dropdown icons/arrows */
[class*="dropdown"] svg,
[data-testid*="dropdown"] svg {
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    min-height: 18px !important;
    max-width: 18px !important;
    max-height: 18px !important;
}

/* Input fields consistent height */
input, select, .wrap-inner {
    min-height: 38px !important;
}
"""

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def atualizar_racas(especie: str):
    """Atualiza dropdown de raças baseado na espécie"""
    if especie in ESPECIES:
        racas = ESPECIES[especie]["racas"]
        return gr.update(choices=racas, value=racas[0])
    return gr.update(choices=[], value=None)

def validar_caso(especie, idade, peso, sintomas):
    """Valida os dados do caso"""
    if not especie:
        return False, "Selecione a espécie"
    if not idade:
        return False, "Indique a idade"
    if not peso:
        return False, "Indique o peso"
    if not sintomas:
        return False, "Descreva os sintomas"
    return True, ""

def criar_resumo_html(especie, raca, idade, peso, sexo, castrado, sintomas):
    """Cria HTML do resumo do caso"""
    icon = ESPECIES.get(especie, {}).get("icon", "🐾")
    castrado_txt = "Sim" if castrado else "Não"
    sintomas_preview = sintomas[:200] + '...' if len(sintomas) > 200 else sintomas
    
    return f"""
    <div class="case-summary">
        <table>
            <tr><td>{icon} Espécie</td><td>{especie}</td></tr>
            <tr><td>Raça</td><td>{raca or 'N/A'}</td></tr>
            <tr><td>Idade</td><td>{idade}</td></tr>
            <tr><td>Peso</td><td>{peso} kg</td></tr>
            <tr><td>Sexo</td><td>{sexo}</td></tr>
            <tr><td>Castrado</td><td>{castrado_txt}</td></tr>
        </table>
        <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #99f6e4;">
            <strong>Sintomas:</strong> {sintomas_preview}
        </div>
    </div>
    """

def formatar_resultado(results: dict, vision_model: str = "", text_model: str = "") -> str:
    """Formata resultado em Markdown"""
    if not results:
        return "❌ Não foi possível gerar o diagnóstico."
    
    md = []
    
    # Performance badge + Model badges
    perf = results.get("performance", {})
    tempo = perf.get("total_ms")
    badges = []
    if tempo:
        segundos = tempo / 1000
        badges.append(f'<span class="perf-badge">⏱️ {segundos:.1f}s</span>')
    if vision_model:
        # Mostrar nome curto do modelo
        vision_short = vision_model.split("/")[-1].split(":")[0]
        badges.append(f'<span class="model-badge">👁️ {vision_short}</span>')
    if text_model:
        text_short = text_model.split("/")[-1].split(":")[0]
        badges.append(f'<span class="model-badge">🧠 {text_short}</span>')
    
    if badges:
        md.append(f'<div style="margin-bottom: 1rem;">{"".join(badges)}</div>\n')
    
    # Análise Visual
    visual = results.get("visual_analysis", "")
    if visual and visual != "Nenhuma imagem fornecida":
        md.append("\n### 🔍 Análise Visual\n")
        md.append(visual)
        md.append("\n\n---\n\n")
    
    # Diagnóstico
    if results.get("diagnosis"):
        md.append("### 🩺 Diagnóstico e Recomendações\n")
        md.append(results["diagnosis"])
        md.append("\n\n")
    
    # Pesquisa Web
    if results.get("research"):
        md.append("---\n\n### 📚 Informação Complementar\n")
        md.append(results["research"])
        md.append("\n")
    
    # Disclaimer final
    md.append("""
<div class="final-disclaimer">
    <span class="icon">⚠️</span>
    <div>
        <strong>Lembrete:</strong> Este relatório é gerado por IA e serve apenas como apoio.
        Consulte sempre um médico veterinário para diagnóstico definitivo e tratamento.
    </div>
</div>
""")
    
    return "".join(md)

# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def executar_diagnostico(
    especie, raca, idade, peso, sexo, castrado, historico, sintomas_texto,
    sintomas_gerais, sintomas_digestivos, sintomas_respiratorios, sintomas_pele, sintomas_movimento,
    images, urgencia, vision_model, text_model, progress=gr.Progress()
):
    """Executa o diagnóstico completo"""
    
    # Validação
    valido, erro = validar_caso(especie, idade, peso, sintomas_texto)
    if not valido:
        return "", f"### ⚠️ {erro}\n\nPor favor, preencha todos os campos obrigatórios.", ""
    
    # Combinar sintomas
    todos_sintomas = []
    for lista in [sintomas_gerais, sintomas_digestivos, sintomas_respiratorios, sintomas_pele, sintomas_movimento]:
        if lista:
            todos_sintomas.extend(lista)
    
    sintomas_final = sintomas_texto
    if todos_sintomas:
        sintomas_final += f". Sintomas selecionados: {', '.join(todos_sintomas)}"
    
    # Mapear urgência (remover emojis para o backend)
    mapa_urgencia = {
        "🟢 Rotina": "Rotina",
        "🟡 Moderada": "Moderada",
        "🔴 Urgente": "Urgente"
    }
    nivel_urgencia = mapa_urgencia.get(urgencia, "Rotina")
    
    # Obter IDs dos modelos selecionados
    vision_model_id = VISION_MODELS.get(vision_model, settings.VLM_MODEL)
    text_model_id = TEXT_MODELS.get(text_model, settings.LLM_OPENROUTER_1)
    
    try:
        progress(0.1, desc="Preparando análise...")
        
        case = CaseInput(
            especie=especie,
            raca=raca or "",
            idade=idade,
            peso=peso,
            sexo=sexo,
            castrado=castrado,
            historico=historico or "",
            sintomas=sintomas_final,
            urgencia=nivel_urgencia,
            image_paths=[images] if images else []
        )
        
        progress(0.3, desc="Analisando dados clínicos...")
        # Passar modelos selecionados para o orchestrator
        results = vet_system.run_diagnosis(case, vision_model=vision_model_id, text_model=text_model_id)
        
        progress(0.9, desc="Gerando relatório...")
        
        resumo = criar_resumo_html(especie, raca, idade, peso, sexo, castrado, sintomas_final)
        diagnostico = formatar_resultado(results, vision_model_id, text_model_id)
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        
        progress(1.0, desc="Concluído!")
        
        return resumo, diagnostico, json_str
        
    except Exception as e:
        return "", f"### ❌ Erro\n\n```\n{str(e)}\n```", ""

# ============================================================
# INTERFACE
# ============================================================

def criar_interface():
    """Cria interface Gradio"""
    
    with gr.Blocks(title="🐾 Vet Agents") as app:
        
        # Injetar CSS
        gr.HTML(f"<style>{CSS}</style>")
        
        # Header
        gr.HTML("""
        <div class="app-header">
            <h1>🐾 Vet Agents</h1>
            <p>Sistema inteligente de apoio ao diagnóstico veterinário</p>
        </div>
        """)
        
        # Disclaimer
        gr.HTML("""
        <div class="disclaimer">
            <strong>⚠️ Aviso:</strong> Este sistema é uma ferramenta de apoio e não substitui 
            a consulta presencial com um médico veterinário qualificado.
        </div>
        """)
        
        with gr.Row():
            # === COLUNA ESQUERDA: FORMULÁRIO ===
            with gr.Column(scale=1):
                
                gr.HTML('<div class="section-title">📋 Dados do Animal</div>')
                
                with gr.Row():
                    especie = gr.Dropdown(
                        choices=list(ESPECIES.keys()),
                        label="Espécie",
                        value="Cão"
                    )
                    raca = gr.Dropdown(
                        choices=ESPECIES["Cão"]["racas"],
                        label="Raça",
                        value="SRD (Sem Raça Definida)"
                    )
                
                with gr.Row():
                    idade = gr.Textbox(label="Idade", placeholder="Ex: 5 anos")
                    peso = gr.Textbox(label="Peso (kg)", placeholder="Ex: 12")
                
                with gr.Row():
                    sexo = gr.Dropdown(
                        choices=["Macho", "Fêmea", "Desconhecido"],
                        label="Sexo",
                        value="Desconhecido"
                    )
                    castrado = gr.Checkbox(label="Castrado/Esterilizado")
                
                historico = gr.Textbox(
                    label="Histórico Médico",
                    placeholder="Vacinas, alergias, doenças anteriores...",
                    lines=2
                )
                
                gr.HTML('<div class="section-title" style="margin-top: 1rem;">🩺 Sintomas</div>')
                
                sintomas_texto = gr.Textbox(
                    label="Descrição dos sintomas",
                    placeholder="Descreva detalhadamente o que observou...",
                    lines=4
                )
                
                with gr.Accordion("➕ Selecionar sintomas comuns", open=False):
                    sintomas_gerais = gr.CheckboxGroup(choices=SINTOMAS["Gerais"], label="Gerais")
                    sintomas_digestivos = gr.CheckboxGroup(choices=SINTOMAS["Digestivos"], label="Digestivos")
                    sintomas_respiratorios = gr.CheckboxGroup(choices=SINTOMAS["Respiratórios"], label="Respiratórios")
                    sintomas_pele = gr.CheckboxGroup(choices=SINTOMAS["Pele"], label="Pele")
                    sintomas_movimento = gr.CheckboxGroup(choices=SINTOMAS["Movimento"], label="Movimento")
                
                gr.HTML('<div class="section-title" style="margin-top: 1rem;">📷 Imagens</div>')
                
                images = gr.Image(
                    label="Upload de imagem (opcional)",
                    type="filepath"
                )
                
                urgencia = gr.Radio(
                    choices=["🟢 Rotina", "🟡 Moderada", "🔴 Urgente"],
                    label="Nível de urgência",
                    value="🟢 Rotina"
                )
                
                # === CONFIGURAÇÃO DE MODELOS ===
                with gr.Accordion("⚙️ Configuração de Modelos IA", open=False):
                    gr.HTML('<div class="config-section">')
                    with gr.Row():
                        vision_model = gr.Dropdown(
                            choices=list(VISION_MODELS.keys()),
                            label="👁️ Modelo de Visão (imagens)",
                            value="Gemini Flash (Google)",
                            info="Usado para analisar imagens"
                        )
                        text_model = gr.Dropdown(
                            choices=list(TEXT_MODELS.keys()),
                            label="🧠 Modelo de Texto (diagnóstico)",
                            value="Grok 4.1 Fast (xAI)",
                            info="Usado para gerar o diagnóstico"
                        )
                    gr.HTML('</div>')
                
                btn_diagnostico = gr.Button(
                    "🔍 Analisar Caso",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"]
                )
            
            # === COLUNA DIREITA: RESULTADOS ===
            with gr.Column(scale=1):
                
                gr.HTML('<div class="results-header">📊 Resultado da Análise</div>')
                
                resumo_html = gr.HTML()
                diagnostico_md = gr.Markdown()
                
                # Separador visual antes do accordion
                gr.HTML('<div style="margin-top: 1rem;"></div>')
                
                with gr.Accordion("🔧 Dados técnicos (JSON)", open=False):
                    json_output = gr.Code(language="json")
        
        # Footer
        gr.HTML("""
        <div class="app-footer">
            <p>🐾 Vet Agents © 2025 • Sistema de apoio ao diagnóstico veterinário com IA</p>
        </div>
        """)
        
        # === EVENT HANDLERS ===
        especie.change(
            fn=atualizar_racas,
            inputs=[especie],
            outputs=[raca]
        )
        
        btn_diagnostico.click(
            fn=executar_diagnostico,
            inputs=[
                especie, raca, idade, peso, sexo, castrado, historico, sintomas_texto,
                sintomas_gerais, sintomas_digestivos, sintomas_respiratorios, sintomas_pele, sintomas_movimento,
                images, urgencia, vision_model, text_model
            ],
            outputs=[resumo_html, diagnostico_md, json_output]
        )
    
    return app

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app = criar_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True
    )