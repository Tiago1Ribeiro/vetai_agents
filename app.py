# app.py - Interface Streamlit Clean & Profissional
# ============================================================

# Suprimir warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*duckduckgo.*")
warnings.filterwarnings("ignore", message=".*LangChain.*")

import streamlit as st
from datetime import datetime
from typing import Optional
import json
import tempfile
import os

from agents.orchestrator import VetDiagnosisOrchestrator, CaseInput
from config.settings import settings

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="🐾 Vet Agents",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS MODERNO (cores corrigidas para a11y)
# ============================================================
st.markdown("""
<style>
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
.stApp {
    background: var(--bg) !important;
}

.main .block-container {
    max-width: 1200px !important;
    padding-top: 1rem !important;
}

/* === HEADER === */
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

/* === DISCLAIMER === */
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

/* === SECTION TITLES === */
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

/* === RESULTS HEADER === */
.results-header {
    background: linear-gradient(135deg, var(--primary) 0%, #0f766e 100%);
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    font-weight: 600;
    font-size: 1.1rem;
}

/* === CASE SUMMARY === */
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

/* === BADGES === */
.badges-container {
    margin-bottom: 1rem;
}

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
    margin-right: 0.5rem;
}

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
    margin-right: 0.5rem;
}

/* === FINAL DISCLAIMER === */
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

/* === FOOTER === */
.app-footer {
    text-align: center;
    padding: 1.5rem;
    color: var(--text-light);
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}

/* === WAITING MESSAGE === */
.waiting-message {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid #7dd3fc;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #0369a1;
}

.waiting-message h3 {
    margin: 0 0 0.5rem 0;
    color: #0c4a6e;
}

/* === STREAMLIT OVERRIDES === */
.stButton > button {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3) !important;
}

/* Fix selectbox height */
.stSelectbox > div > div {
    min-height: 42px !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: var(--text) !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Progress bar color */
.stProgress > div > div > div > div {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURAÇÕES / DADOS
# ============================================================

# Modelos disponíveis para Visão (VLM)
VISION_MODELS = {
    "Gemini 2.5 Flash (Google)": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite (Google)": "gemini-2.5-flash-lite",
    "Pixtral 12B (Mistral)": "pixtral-12b-2409",
}

# Modelos disponíveis para Texto/Diagnóstico (LLM)
TEXT_MODELS = {
    "Grok 4.1 Fast (xAI) - Free": "x-ai/grok-4.1-fast:free",
    "Gemma 3 27B (Google) - Free": "google/gemma-3-27b-it:free",
    "DeepSeek R1 Chimera - Free": "tngtech/deepseek-r1t-chimera:free",
    "GLM 4.5 Air (Zhipu) - Free": "z-ai/glm-4.5-air:free",
    "Mistral Small (Mistral)": "mistral-small-latest",
}

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
# INICIALIZAÇÃO DO SISTEMA
# ============================================================

@st.cache_resource
def get_vet_system():
    """Inicializa o sistema de diagnóstico (cached)"""
    return VetDiagnosisOrchestrator()

vet_system = get_vet_system()

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def validar_caso(especie: str, idade: str, peso: str, sintomas: str) -> tuple[bool, str]:
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


def criar_resumo_html(especie: str, raca: str, idade: str, peso: str, 
                      sexo: str, castrado: bool, sintomas: str) -> str:
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


def criar_badges_html(tempo_ms: Optional[float], vision_model: str, text_model: str) -> str:
    """Cria HTML dos badges de performance e modelos"""
    badges = []
    
    if tempo_ms:
        segundos = tempo_ms / 1000
        badges.append(f'<span class="perf-badge">⏱️ {segundos:.1f}s</span>')
    
    if vision_model:
        vision_short = vision_model.split("/")[-1].split(":")[0]
        badges.append(f'<span class="model-badge">👁️ {vision_short}</span>')
    
    if text_model:
        text_short = text_model.split("/")[-1].split(":")[0]
        badges.append(f'<span class="model-badge">🧠 {text_short}</span>')
    
    if badges:
        return f'<div class="badges-container">{"".join(badges)}</div>'
    return ""


def processar_imagem(uploaded_file) -> Optional[str]:
    """Processa imagem uploaded e retorna path temporário"""
    if uploaded_file is None:
        return None
    
    # Criar arquivo temporário
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def executar_diagnostico(
    especie: str, raca: str, idade: str, peso: str, sexo: str, 
    castrado: bool, historico: str, sintomas_texto: str,
    sintomas_selecionados: list, image_path: Optional[str],
    urgencia: str, vision_model_id: str, text_model_id: str,
    progress_callback
) -> dict:
    """Executa o diagnóstico completo"""
    
    # Combinar sintomas
    sintomas_final = sintomas_texto
    if sintomas_selecionados:
        sintomas_final += f". Sintomas selecionados: {', '.join(sintomas_selecionados)}"
    
    # Mapear urgência (remover emojis)
    mapa_urgencia = {
        "🟢 Rotina": "Rotina",
        "🟡 Moderada": "Moderada", 
        "🔴 Urgente": "Urgente"
    }
    nivel_urgencia = mapa_urgencia.get(urgencia, "Rotina")
    
    progress_callback(10, "Preparando análise...")
    
    # Criar caso
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
        image_paths=[image_path] if image_path else []
    )
    
    progress_callback(30, "Analisando dados clínicos...")
    
    # Executar diagnóstico
    results = vet_system.run_diagnosis(
        case, 
        vision_model=vision_model_id, 
        text_model=text_model_id
    )
    
    progress_callback(90, "Gerando relatório...")
    
    return {
        "results": results,
        "sintomas_final": sintomas_final,
        "vision_model_id": vision_model_id,
        "text_model_id": text_model_id
    }


# ============================================================
# INICIALIZAÇÃO DO SESSION STATE
# ============================================================

if "resultado" not in st.session_state:
    st.session_state.resultado = None
if "processando" not in st.session_state:
    st.session_state.processando = False

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="app-header">
    <h1>🐾 Vet Agents</h1>
    <p>Sistema inteligente de apoio ao diagnóstico veterinário</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Aviso:</strong> Este sistema é uma ferramenta de apoio e não substitui 
    a consulta presencial com um médico veterinário qualificado.
</div>
""", unsafe_allow_html=True)

# ============================================================
# DEBUG: VERIFICAR API KEYS (sidebar)
# ============================================================
with st.sidebar:
    st.markdown("### 🔑 Estado das API Keys")
    
    # Google API Key
    google_key = settings.GOOGLE_API_KEY
    if google_key and len(google_key) > 10:
        st.success(f"✅ Google: {google_key[:8]}...{google_key[-4:]}")
    else:
        st.error("❌ Google: NÃO CONFIGURADA")
    
    # OpenRouter API Key
    openrouter_key = settings.OPENROUTER_API_KEY
    if openrouter_key and len(openrouter_key) > 10:
        st.success(f"✅ OpenRouter: {openrouter_key[:8]}...{openrouter_key[-4:]}")
    else:
        st.error("❌ OpenRouter: NÃO CONFIGURADA")
    
    # Mistral API Key
    mistral_key = settings.MISTRAL_API_KEY
    if mistral_key and len(mistral_key) > 10:
        st.success(f"✅ Mistral: {mistral_key[:8]}...{mistral_key[-4:]}")
    else:
        st.warning("⚠️ Mistral: não configurada (opcional)")
    
    st.markdown("---")
    st.caption("Expande a sidebar para ver o estado das chaves API")

# ============================================================
# LAYOUT PRINCIPAL
# ============================================================

col_form, col_results = st.columns([1, 1], gap="large")

# === COLUNA ESQUERDA: FORMULÁRIO ===
with col_form:
    
    # === CONFIGURAÇÃO DE MODELOS (NO TOPO, SEMPRE VISÍVEL) ===
    st.markdown('<div class="section-title">⚙️ Configuração de Modelos IA</div>', unsafe_allow_html=True)
    col_vm, col_tm = st.columns(2)
    with col_vm:
        vision_model = st.selectbox(
            "👁️ Modelo de Visão (imagens)",
            options=list(VISION_MODELS.keys()),
            index=0,
            help="Usado para analisar imagens",
            key="vision_model"
        )
    with col_tm:
        text_model = st.selectbox(
            "🧠 Modelo de Texto (diagnóstico)",
            options=list(TEXT_MODELS.keys()),
            index=0,
            help="Usado para gerar o diagnóstico",
            key="text_model"
        )
    
    st.markdown('<div class="section-title" style="margin-top: 1rem;">📋 Dados do Animal</div>', unsafe_allow_html=True)
    
    # Espécie e Raça
    col_esp, col_raca = st.columns(2)
    with col_esp:
        especie = st.selectbox(
            "Espécie",
            options=list(ESPECIES.keys()),
            index=0,
            key="especie"
        )
    with col_raca:
        # Atualizar raças dinamicamente baseado na espécie
        racas_disponiveis = ESPECIES[especie]["racas"]
        raca = st.selectbox(
            "Raça",
            options=racas_disponiveis,
            index=0,
            key="raca"
        )
    
    # Idade e Peso
    col_idade, col_peso = st.columns(2)
    with col_idade:
        idade = st.text_input("Idade", placeholder="Ex: 5 anos", key="idade")
    with col_peso:
        peso = st.text_input("Peso (kg)", placeholder="Ex: 12", key="peso")
    
    # Sexo e Castrado
    col_sexo, col_cast = st.columns(2)
    with col_sexo:
        sexo = st.selectbox(
            "Sexo",
            options=["Macho", "Fêmea", "Desconhecido"],
            index=2,
            key="sexo"
        )
    with col_cast:
        st.write("")  # Spacing
        st.write("")  # Spacing  
        castrado = st.checkbox("Castrado/Esterilizado", key="castrado")
    
    # Histórico
    historico = st.text_area(
        "Histórico Médico",
        placeholder="Vacinas, alergias, doenças anteriores...",
        height=68,
        key="historico"
    )
    
    # === SINTOMAS ===
    st.markdown('<div class="section-title" style="margin-top: 1rem;">🩺 Sintomas</div>', unsafe_allow_html=True)
    
    sintomas_texto = st.text_area(
        "Descrição dos sintomas",
        placeholder="Descreva detalhadamente o que observou...",
        height=100,
        key="sintomas_texto"
    )
    
    # Sintomas comuns (expansível)
    with st.expander("➕ Selecionar sintomas comuns"):
        sintomas_selecionados = []
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sel_gerais = st.multiselect("Gerais", SINTOMAS["Gerais"], key="sint_gerais")
            sel_digestivos = st.multiselect("Digestivos", SINTOMAS["Digestivos"], key="sint_digestivos")
            sel_respiratorios = st.multiselect("Respiratórios", SINTOMAS["Respiratórios"], key="sint_respiratorios")
        with col_s2:
            sel_pele = st.multiselect("Pele", SINTOMAS["Pele"], key="sint_pele")
            sel_movimento = st.multiselect("Movimento", SINTOMAS["Movimento"], key="sint_movimento")
        
        sintomas_selecionados = sel_gerais + sel_digestivos + sel_respiratorios + sel_pele + sel_movimento
    
    # === IMAGENS ===
    st.markdown('<div class="section-title" style="margin-top: 1rem;">📷 Imagens</div>', unsafe_allow_html=True)
    
    # Imagens de exemplo
    st.markdown("**Imagens de exemplo para teste:**")
    example_images = {
        "🐱 Gato - Uveíte": "images/cat-with-Uveitis-compressed.jpg",
        "🐱 Gato - Prolapso 3ª Pálpebra": "images/cat-with-Prolapsed-Third-Eyelids-compressed.jpg",
        "🐱 Gato - Conjuntivite": "images/cat-with-infectious-conjunctivitis-compressed.jpg",
        "🐱 Gato - Cataratas": "images/adorable-fluffy-Siberian-with-cataracts-compressed.jpg",
        "🐕 Cão - Dermatite Atópica": "images/Dermatite_atpica.jpg",
    }
    
    col_ex1, col_ex2 = st.columns(2)
    selected_example = None
    
    with col_ex1:
        example_choice = st.selectbox(
            "Selecionar imagem de exemplo",
            options=["Nenhuma"] + list(example_images.keys()),
            key="example_image"
        )
    
    # Mostrar preview da imagem de exemplo
    if example_choice != "Nenhuma":
        example_path = example_images[example_choice]
        if os.path.exists(example_path):
            selected_example = example_path
            st.image(example_path, caption=f"Exemplo: {example_choice}", use_container_width=True)
    
    uploaded_image = st.file_uploader(
        "Ou faça upload da sua imagem",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload"
    )
    
    if uploaded_image:
        st.image(uploaded_image, caption="Imagem carregada", use_container_width=True)
    
    # Urgência
    urgencia = st.radio(
        "Nível de urgência",
        options=["🟢 Rotina", "🟡 Moderada", "🔴 Urgente"],
        horizontal=True,
        key="urgencia"
    )
    
    # === BOTÃO DE ANÁLISE ===
    st.write("")  # Spacing
    btn_analisar = st.button("🔍 Analisar Caso", type="primary", use_container_width=True)


# === COLUNA DIREITA: RESULTADOS ===
with col_results:
    
    st.markdown('<div class="results-header">📊 Resultado da Análise</div>', unsafe_allow_html=True)
    
    # Container para resultados
    resultado_container = st.container()


# ============================================================
# PROCESSAMENTO (após o layout)
# ============================================================

if btn_analisar:
    # Validação
    valido, erro = validar_caso(especie, idade, peso, sintomas_texto)
    
    if not valido:
        with col_results:
            st.error(f"⚠️ {erro}")
            st.warning("Por favor, preencha todos os campos obrigatórios.")
    else:
        # Processar imagem - priorizar upload, depois exemplo
        image_path = processar_imagem(uploaded_image)
        
        # Se não há upload mas há exemplo selecionado, usar exemplo
        if not image_path and 'example_image' in st.session_state:
            example_choice = st.session_state.example_image
            if example_choice != "Nenhuma":
                example_images = {
                    "🐱 Gato - Uveíte": "images/cat-with-Uveitis-compressed.jpg",
                    "🐱 Gato - Prolapso 3ª Pálpebra": "images/cat-with-Prolapsed-Third-Eyelids-compressed.jpg",
                    "🐱 Gato - Conjuntivite": "images/cat-with-infectious-conjunctivitis-compressed.jpg",
                    "🐱 Gato - Cataratas": "images/adorable-fluffy-Siberian-with-cataracts-compressed.jpg",
                    "🐕 Cão - Dermatite Atópica": "images/Dermatite_atpica.jpg",
                }
                example_path = example_images.get(example_choice)
                if example_path and os.path.exists(example_path):
                    image_path = example_path
        
        # Obter IDs dos modelos
        vision_model_id = VISION_MODELS.get(vision_model, settings.VLM_MODEL)
        text_model_id = TEXT_MODELS.get(text_model, settings.LLM_OPENROUTER_1)
        
        # Coletar sintomas selecionados
        todos_sintomas = []
        if 'sint_gerais' in st.session_state:
            todos_sintomas.extend(st.session_state.sint_gerais)
        if 'sint_digestivos' in st.session_state:
            todos_sintomas.extend(st.session_state.sint_digestivos)
        if 'sint_respiratorios' in st.session_state:
            todos_sintomas.extend(st.session_state.sint_respiratorios)
        if 'sint_pele' in st.session_state:
            todos_sintomas.extend(st.session_state.sint_pele)
        if 'sint_movimento' in st.session_state:
            todos_sintomas.extend(st.session_state.sint_movimento)
        
        with col_results:
            # Progress bar
            progress_bar = st.progress(0, text="Iniciando análise...")
            status_text = st.empty()
            
            def update_progress(value: int, text: str):
                progress_bar.progress(value, text=text)
            
            try:
                # Executar diagnóstico
                resultado = executar_diagnostico(
                    especie=especie,
                    raca=raca,
                    idade=idade,
                    peso=peso,
                    sexo=sexo,
                    castrado=castrado,
                    historico=historico,
                    sintomas_texto=sintomas_texto,
                    sintomas_selecionados=todos_sintomas,
                    image_path=image_path,
                    urgencia=urgencia,
                    vision_model_id=vision_model_id,
                    text_model_id=text_model_id,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(100, text="Concluído! ✅")
                
                # Guardar no session state
                st.session_state.resultado = resultado
                st.session_state.resultado["especie"] = especie
                st.session_state.resultado["raca"] = raca
                st.session_state.resultado["idade"] = idade
                st.session_state.resultado["peso"] = peso
                st.session_state.resultado["sexo"] = sexo
                st.session_state.resultado["castrado"] = castrado
                
                # Limpar progress após sucesso
                import time
                time.sleep(0.5)
                progress_bar.empty()
                
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Erro durante a análise")
                st.code(str(e), language="text")
                st.session_state.resultado = None


# ============================================================
# EXIBIÇÃO DE RESULTADOS
# ============================================================

with col_results:
    if st.session_state.resultado:
        resultado = st.session_state.resultado
        results = resultado["results"]
        
        # Resumo do caso
        st.markdown(criar_resumo_html(
            resultado["especie"],
            resultado["raca"],
            resultado["idade"],
            resultado["peso"],
            resultado["sexo"],
            resultado["castrado"],
            resultado["sintomas_final"]
        ), unsafe_allow_html=True)
        
        # Badges de performance e modelos
        perf = results.get("performance", {})
        tempo_ms = perf.get("total_ms")
        st.markdown(criar_badges_html(
            tempo_ms,
            resultado["vision_model_id"],
            resultado["text_model_id"]
        ), unsafe_allow_html=True)
        
        # Análise Visual
        visual = results.get("visual_analysis", "")
        if visual and visual != "Nenhuma imagem fornecida":
            st.markdown("### 🔍 Análise Visual")
            st.markdown(visual)
            st.divider()
        
        # Diagnóstico
        if results.get("diagnosis"):
            st.markdown("### 🩺 Diagnóstico e Recomendações")
            st.markdown(results["diagnosis"])
        
        # Pesquisa Web
        st.divider()
        st.markdown("### 📚 Informação Complementar (Web)")
        
        # Mostrar info da pesquisa
        knowledge_info = results.get("knowledge_gathered", {})
        web_chars = knowledge_info.get("web_results_chars", 0)
        
        if results.get("research") and len(results.get("research", "")) > 50:
            st.markdown(results["research"])
        elif web_chars > 0:
            st.info(f"Pesquisa web realizada ({web_chars} caracteres encontrados)")
        else:
            st.info("🔍 Pesquisa web não retornou resultados relevantes para este caso.")
        
        # Disclaimer final
        st.markdown("""
        <div class="final-disclaimer">
            <span class="icon">⚠️</span>
            <div>
                <strong>Lembrete:</strong> Este relatório é gerado por IA e serve apenas como apoio.
                Consulte sempre um médico veterinário para diagnóstico definitivo e tratamento.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Dados técnicos (JSON)
        with st.expander("🔧 Dados técnicos (JSON)"):
            st.json(results)
    
    elif not btn_analisar:
        # Mensagem inicial quando não há resultado
        st.markdown("""
        <div class="waiting-message">
            <h3>👈 Preencha o formulário</h3>
            <p>Complete os dados do animal e clique em "Analisar Caso" para obter o diagnóstico.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="app-footer">
    <p>🐾 Vet Agents © 2025 • Sistema de apoio ao diagnóstico veterinário com IA</p>
</div>
""", unsafe_allow_html=True)
