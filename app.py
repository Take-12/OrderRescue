import streamlit as st
import sqlite3
import pandas as pd
import pickle
import os
import time
import requests
import json
from gtts import gTTS
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────
# DESCARGA AUTOMÁTICA DE LA BASE DE DATOS (PARA STREAMLIT CLOUD)
# ─────────────────────────────────────────────────────────────
def descargar_base_de_datos_si_falta():
    ruta_db = "order_rescue.db"
    file_id = "1RlpSLZkkxkZljczo3GyCvvGxmS90-cCw" # ID del archivo de Google Drive del usuario
    
    if not os.path.exists(ruta_db):
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            with st.spinner("⏬ Descargando base de datos desde Google Drive para el primer inicio... (esto puede tomar 1 o 2 minutos)"):
                gdown.download(url, ruta_db, quiet=False)
            st.success("✅ Base de datos descargada con éxito.")
        except Exception as e:
            st.error(f"❌ Error al descargar la base de datos: {e}")
            st.info("Asegúrate de que el enlace de Google Drive tenga permisos para 'Cualquier persona con el enlace'.")
            st.stop()

# Ejecutar descarga si es necesario (ej. en Streamlit Cloud)
descargar_base_de_datos_si_falta()

# Importamos la función de optimización bayesiana
from optimizacion_bayesiana import ejecutar_optimizacion_bayesiana
from smart_order_rescue import render_smart_order_rescue

# Función para interactuar con la API de Gemini 2.5 Flash
def llamar_api_gemini(mensaje_usuario, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    contents = []
    # Cargar los últimos 8 mensajes para mantener memoria de la conversación
    for msg in st.session_state.chatbot_historial[-8:]:
        role_map = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role_map,
            "parts": [{"text": msg["content"]}]
        })
        
    contents.append({
        "role": "user",
        "parts": [{"text": mensaje_usuario}]
    })
    
    system_instruction = (
        "Eres el Asistente Virtual Inteligente de Arca Continental para la plataforma 'Smart Order Rescue'. "
        "Tu misión es guiar de manera formal, clara y empática a los usuarios (vendedores, ejecutivos comerciales y operadores de almacén) "
        "sobre el funcionamiento de la aplicación.\n\n"
        "INFORMACIÓN CLAVE DEL SIMULADOR QUE DEBES EXPLICAR:\n"
        "1. Roles de usuario (en la barra lateral):\n"
        "   - Operador de CEDI (Admin): Tiene la 'Bandeja de Entrada' (alertas de stock y priorización automatizada por gravedad de déficit), "
        "     'Confirmación de Descuentos' (slider de compensación de lealtad de 0% a 5%) y el 'Dashboard CEDI y Analíticas' (KPIs históricos del CEDI).\n"
        "   - Comprador B2B (Cliente): Portal de pedidos. Credenciales de prueba: centro/centro123 (Restaurante Centro), pastor/pastor123 (Taquería El Pastor), "
        "     esquina/esquina123 (Abarrotes La Esquina), fitzone/fitzone123 (Gimnasio FitZone), smart/smart123 (Supermercado Smart). "
        "     Si el pedido tiene 10% de proximidad o excede el stock, salta una advertencia y solicita pre-autorizar alternativas a cambio de un descuento.\n"
        "2. Pestañas de Smart Order Rescue (para ejecutivos comerciales):\n"
        "   - Predicción XGBoost: Usa Machine Learning (modelo_xgboost.pkl) para calcular la probabilidad de sustitución (Verde/Amarillo/Rojo).\n"
        "   - Demanda Prophet: Pronostica a 7 días la demanda estacional por CEDI y producto usando regresión lineal.\n"
        "   - Recomendador de Sustitutos: Sugiere el reemplazo de producto ideal y da una plantilla de mensaje comercial formal.\n"
        "   - Simulador CEDI (Digital Twin): Proyecta impactos de mermas/demanda y sugiere traslados de stock inter-CEDI.\n"
        "   - Base de Datos SQL: Navegador visual de registros históricos de tablas (pedidos, sustituciones, clientes, etc.).\n\n"
        "Responde siempre en español, de forma breve, estructurada y profesional."
    )
    
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 350
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            try:
                return data['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                return "Error: Respuesta de Gemini en formato no esperado."
        else:
            err_msg = response.text
            if "API_KEY_INVALID" in err_msg:
                return "❌ La API Key de Gemini ingresada es inválida. Por favor, verifícala."
            return f"❌ Error de Gemini API (Código {response.status_code}): {err_msg[:150]}"
    except Exception as e:
        return f"❌ Error de conexión: {e}"




# Configuración de página
st.set_page_config(
    page_title="Smart Order Rescue - Arca Continental",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para apariencia premium
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #e41e26;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #e41e26;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .chat-bubble {
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        max-width: 75%;
        color: #222222;
    }
    .chat-agent {
        background-color: #ffebe6;
        color: #222222;
        border-left: 4px solid #e41e26;
        margin-right: auto;
    }
    .chat-user {
        background-color: #e0f2f1;
        color: #222222;
        border-right: 4px solid #009688;
        margin-left: auto;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Directorios y rutas
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "order_rescue.db")
model_path = os.path.join(current_dir, "modelo_order_rescue.pkl")

# Conexión a base de datos
def get_db_connection():
    return sqlite3.connect(db_path)

# Cargar la lista de CEDIS
@st.cache_data
def get_lista_cedis():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT DISTINCT cedis FROM orders WHERE cedis != '' ORDER BY cedis;", conn)
    conn.close()
    return df['cedis'].tolist()

# Obtener los top 5 productos de un CEDI y sus demandas promedio
@st.cache_data
def get_top_productos_cedi(cedi):
    conn = get_db_connection()
    query = """
        SELECT 
            d.nombre_sku_solicitado as producto,
            AVG(d.quantity) as avg_qty,
            COUNT(*) as total_pedidos
        FROM order_details d
        JOIN orders o ON d.id_pedido = o.id_pedido
        WHERE o.cedis = ? AND d.nombre_sku_solicitado IS NOT NULL AND d.nombre_sku_solicitado != 'None' AND d.nombre_sku_solicitado != ''
        GROUP BY d.nombre_sku_solicitado
        ORDER BY total_pedidos DESC
        LIMIT 5;
    """
    df = pd.read_sql_query(query, conn, params=(cedi,))
    conn.close()
    
    if len(df) == 0:
        return pd.DataFrame({
            "producto": ["Coca - Cola", "Ciel Agua Purificada", "Powerade Moras", "Topo Chico Agua Mineral", "Yogurt Toni Durazno 110 Gr."],
            "avg_qty": [30.0, 15.0, 10.0, 12.0, 8.0]
        })
    return df

# Cargamos el modelo ML
@st.cache_resource
def load_ml_model():
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_ml_model()

# Función para llamar a ElevenLabs Text-to-Speech API
def generar_voz_elevenlabs(texto, api_key):
    voice_id = "21m00Tcm4TlvDq8ikWAM" 
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": texto,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Error de ElevenLabs API ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Error de conexión con ElevenLabs: {e}")
        return None

# Inicialización de estados de operador
if 'pedido_procesado' not in st.session_state:
    st.session_state.pedido_procesado = False
    st.session_state.producto = ""
    st.session_state.cantidad = 0
    st.session_state.prob_real = 0.0
    st.session_state.mejor_accion = ""
    st.session_state.menor_costo = 0.0
    st.session_state.df_mpc = None
    st.session_state.sustituto_sugerido = ""
    st.session_state.simular_llamada_clic = False

# Inicialización de la bandeja de entrada del operador y descuentos
if 'b2b_alertas_operador' not in st.session_state:
    st.session_state.b2b_alertas_operador = [
        {
            "cliente": "Taquería El Pastor",
            "producto": "Coca - Cola",
            "cantidad": 35,
            "stock_actual": 15,
            "tipo_caso": "excede",
            "fecha": "00:15:30",
            "leido": False
        },
        {
            "cliente": "Restaurante Centro",
            "producto": "Powerade Moras",
            "cantidad": 19,
            "stock_actual": 20,
            "tipo_caso": "cercano",
            "fecha": "00:20:45",
            "leido": False
        },
        {
            "cliente": "Abarrotes La Esquina",
            "producto": "Sprite Lima Limón",
            "cantidad": 25,
            "stock_actual": 17,
            "tipo_caso": "excede",
            "fecha": "00:22:10",
            "leido": True
        }
    ]

if 'descuentos_manuales' not in st.session_state:
    st.session_state.descuentos_manuales = {
        "Restaurante Centro": 4.5,
        "Abarrotes La Esquina": 3.5,
        "Gimnasio FitZone": 5.0,
        "Supermercado Smart": 5.0,
        "Taquería El Pastor": 4.0
    }

if 'descuentos_confirmados' not in st.session_state:
    st.session_state.descuentos_confirmados = {
        "Restaurante Centro": False,
        "Abarrotes La Esquina": False,
        "Gimnasio FitZone": False,
        "Supermercado Smart": False,
        "Taquería El Pastor": False
    }

# Inicialización de estados de comprador B2B
if 'b2b_logged_in' not in st.session_state:
    st.session_state.b2b_logged_in = False
    st.session_state.b2b_user = ""
    st.session_state.b2b_cliente_nombre = ""

if 'b2b_carrito' not in st.session_state:
    st.session_state.b2b_carrito = []

if 'b2b_compra_finalizada' not in st.session_state:
    st.session_state.b2b_compra_finalizada = False

if 'precios_productos' not in st.session_state:
    st.session_state.precios_productos = {
        "Coca - Cola": 1.50,
        "Coca-Cola": 1.50,
        "Coca - Cola Light Sin Cafeína, Botella Pet 600 ml, 24 Piezas": 1.80,
        "Coca-Cola Light Sin Cafeína, Botella Pet 600 ml, 24 Piezas": 1.80,
        "Coca - Cola Sin Azúcar, Botella Pet 1.50 L, 8 Piezas": 2.00,
        "Coca-Cola Sin Azúcar, Botella Pet 1.50 L, 8 Piezas": 2.00,
        "Coca - Cola, Botella Pet 2.00 L Retornable, 8 Piezas": 1.50,
        "Coca-Cola, Botella Pet 2.00 L Retornable, 8 Piezas": 1.50,
        "Coca - Cola Light, Botella Pet 1.50 L, 6 Piezas": 1.80,
        "Coca-Cola Light, Botella Pet 1.50 L, 6 Piezas": 1.80,
        "Coca - Cola, Botella Vidrio 355 ml, 24 Piezas": 1.40,
        "Coca-Cola, Botella Vidrio 355 ml, 24 Piezas": 1.40,
        "Coca - Cola, Botella Pet 400 ml, 12 Piezas": 1.30,
        "Coca-Cola, Botella Pet 400 ml, 12 Piezas": 1.30,
        "Coca - Cola, Botella Vidrio 237 ml, 12 Piezas": 1.10,
        "Coca-Cola, Botella Vidrio 237 ml, 12 Piezas": 1.10,
        "Coca - Cola Light, Botella Pet 1.00 L, 12 Piezas": 1.60,
        "Coca-Cola Light, Botella Pet 1.00 L, 12 Piezas": 1.60,
        "Topo Chico Agua Mineral": 1.20,
        "Topo Chico Agua Mineral, Botella Vidrio 355 ml, 12 Piezas": 1.30,
        "Coca - Cola Zero, Botella Pet 2.00 L, 8 Piezas": 1.70,
        "Coca-Cola Zero, Botella Pet 2.00 L, 8 Piezas": 1.70,
        "Fuze Tea Durazno": 1.30,
        "Fuze Tea Limón, Botella Pet 600 ml, 6 Piezas": 1.40,
        "Coca - Cola Life, Botella Vidrio 500 ml, 24 Piezas": 1.60,
        "Coca-Cola Life, Botella Vidrio 500 ml, 24 Piezas": 1.60,
        "Coca - Cola Light, Botella Vidrio 500 ml, 24 Piezas": 1.60,
        "Coca-Cola Light, Botella Vidrio 500 ml, 24 Piezas": 1.60,
        "Coca - Cola Life, Botella Pet 2.00 L, 8 Piezas": 1.80,
        "Coca-Cola Life, Botella Pet 2.00 L, 8 Piezas": 1.80,
        "Powerade Moras": 1.50,
        "Powerade Uva, Botella Pet 1.00 L, 6 Piezas": 1.60,
        "Sprite Lima Limón": 1.40,
        "Sprite Sin Azúcar Lima Limón, Botella Pet 2.50 L, 8 Piezas": 1.90,
        "Ciel Agua Purificada": 1.00,
        "Ciel Exprim Gasificada Maracuya, Botella Pet 600 ml, 6 Piezas": 1.20,
        "Yogurt Mix Frutilla con Galletas 175 Gr.": 1.10,
        "Yogurt Toni Durazno 110 Gr.": 0.80,
        "Leche Saborizada Toni Frutilla Poma 200 Ml.": 0.90,
        "Leche Saborizada Toni Chocolate Poma 200 Ml.": 0.90,
        "Telefonía Móvil Claro Tarjeta Paquete $5.15": 5.15
    }
    
if 'b2b_pedido_procesado' not in st.session_state:
    st.session_state.b2b_pedido_procesado = False
    st.session_state.b2b_producto = ""
    st.session_state.b2b_cantidad = 0
    st.session_state.b2b_cedi = ""
    st.session_state.b2b_prob_real = 0.0
    st.session_state.b2b_mejor_accion = ""
    st.session_state.b2b_menor_costo = 0.0
    st.session_state.b2b_df_mpc = None
    st.session_state.b2b_sustituto_sugerido = ""
    st.session_state.b2b_simular_llamada_clic = False
    st.session_state.b2b_llamada_confirmada = False
    st.session_state.b2b_proteccion_activada = False
    st.session_state.b2b_regla_10_porciento = False
    st.session_state.b2b_segundo_producto_agregado = False
    st.session_state.b2b_reposicion_opcion_1 = ""
    st.session_state.b2b_reposicion_opcion_2 = ""
    st.session_state.b2b_decision_tomada = ""
    st.session_state.b2b_tipo_caso = ""

# Inicialización de estados de chatbot
if 'chatbot_historial' not in st.session_state:
    st.session_state.chatbot_historial = []
if 'gemini_api_key_temp' not in st.session_state:
    import base64
    st.session_state.gemini_api_key_temp = base64.b64decode("QVEuQWI4Uk42SXJESDZ5NFNHd3JfMlpIRlVMenZwRkE3YmVjelN6V2RkVk1tYTFSNWVubFE=").decode("utf-8")
if 'chat_flotante_abierto' not in st.session_state:
    st.session_state.chat_flotante_abierto = False

# Título Principal
st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>🥤 Smart Order Rescue</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Gemelo Digital e Inteligencia Artificial para la Red de Distribución de Arca Continental</p>", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("<h2 style='color: #e41e26;'>👥 Rol de Usuario</h2>", unsafe_allow_html=True)
rol = st.sidebar.selectbox("Seleccionar Vista", ["🏢 Operador de CEDI (Admin)", "🛒 Comprador B2B (Cliente)"])

lista_cedis = get_lista_cedis()
default_idx = lista_cedis.index("3804") if "3804" in lista_cedis else 0

if rol == "🏢 Operador de CEDI (Admin)":
    st.sidebar.markdown("<h2 style='color: #e41e26;'>📍 Red de Distribución</h2>", unsafe_allow_html=True)
    cedi_seleccionado = st.sidebar.selectbox("Seleccionar CEDI Real", lista_cedis, index=default_idx)
else:
    # En vista comprador, el CEDI se define en la pantalla principal
    if 'b2b_cedi' not in st.session_state or not st.session_state.b2b_cedi:
        st.session_state.b2b_cedi = "3804" if "3804" in lista_cedis else lista_cedis[0]
    cedi_seleccionado = st.session_state.b2b_cedi
    st.sidebar.markdown(f"<h3 style='color: #e41e26;'>📍 CEDI Activo: {cedi_seleccionado}</h3>", unsafe_allow_html=True)

# Resetear estado si cambia CEDI
if 'ultimo_cedi' in st.session_state and st.session_state.ultimo_cedi != cedi_seleccionado:
    st.session_state.pedido_procesado = False
    st.session_state.simular_llamada_clic = False
    st.session_state.b2b_pedido_procesado = False
    st.session_state.b2b_simular_llamada_clic = False
    st.session_state.b2b_llamada_confirmada = False
st.session_state.ultimo_cedi = cedi_seleccionado

df_top_prods = get_top_productos_cedi(cedi_seleccionado)

st.sidebar.markdown("<h2 style='color: #e41e26;'>🏢 Gemelo Digital (Stock)</h2>", unsafe_allow_html=True)
with st.sidebar.expander("📦 Niveles de Inventario", expanded=True):
    st.write(f"Ajustar stock en tiempo real en el **CEDI {cedi_seleccionado}**:")
    inventario = {}
    for _, row in df_top_prods.iterrows():
        prod_name = row['producto']
        avg_demand = int(row['avg_qty'])
        max_slider = max(50, avg_demand * 3)
        default_stock = int(avg_demand * 1.2)
        
        stock_val = st.number_input(
            f"{prod_name} (Promedio: {avg_demand})",
            min_value=0,
            max_value=max_slider,
            value=default_stock,
            step=1,
            key=f"stock_{prod_name}"
        )
        inventario[prod_name] = stock_val

with st.sidebar.expander("💰 Precios de Bebidas", expanded=False):
    st.write("Configurar el costo de cada bebida en tiempo real:")
    for _, row in df_top_prods.iterrows():
        p_name = row['producto']
        precio_actual = st.session_state.precios_productos.get(p_name, 1.50)
        precio_nuevo = st.number_input(
            f"{p_name} ($)",
            min_value=0.10,
            max_value=50.0,
            value=float(precio_actual),
            step=0.05,
            key=f"precio_input_{p_name}"
        )
        st.session_state.precios_productos[p_name] = precio_nuevo

st.sidebar.markdown("---")
st.sidebar.info("El simulador carga dinámicamente los productos estrella basándose en la base de datos SQLite de este CEDI.")

# ----------------- PORTAL COMPRADOR B2B -----------------
if rol == "🛒 Comprador B2B (Cliente)":
    if not st.session_state.b2b_logged_in:
        st.markdown("<h2 style='text-align: center; color: #e41e26;'>🛒 Portal de Pedidos B2B - Arca Continental</h2>", unsafe_allow_html=True)
        st.write("Por favor, inicia sesión con tus credenciales de cliente para realizar pedidos de Arca Continental.")
        
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown("""
            <div style='background-color: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 4px solid #e41e26; color: #222222;'>
                <h3 style='margin-top:0; color:#222222; text-align:center; font-weight: bold;'>🔑 Inicio de Sesión B2B</h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("login_form"):
                usuario = st.text_input("Usuario (ej. centro, pastor, smart)")
                password = st.text_input("Contraseña", type="password")
                entrar = st.form_submit_button("🚀 Entrar al Portal B2B", use_container_width=True)
                
                if entrar:
                    credenciales = {
                        "centro": ("centro123", "Restaurante Centro"),
                        "esquina": ("esquina123", "Abarrotes La Esquina"),
                        "fitzone": ("fitzone123", "Gimnasio FitZone"),
                        "smart": ("smart123", "Supermercado Smart"),
                        "pastor": ("pastor123", "Taquería El Pastor")
                    }
                    
                    if usuario in credenciales and credenciales[usuario][0] == password:
                        st.session_state.b2b_logged_in = True
                        st.session_state.b2b_user = usuario
                        st.session_state.b2b_cliente_nombre = credenciales[usuario][1]
                        
                        # Determinar automáticamente el CEDI más cercano
                        map_cliente_cedi = {
                            "Restaurante Centro": "3804",
                            "Abarrotes La Esquina": "3012",
                            "Gimnasio FitZone": "3803",
                            "Supermercado Smart": "3003",
                            "Taquería El Pastor": "3804"
                        }
                        cliente_nombre = credenciales[usuario][1]
                        cedi_sugerido = map_cliente_cedi.get(cliente_nombre, "3804")
                        if cedi_sugerido not in lista_cedis:
                            cedi_sugerido = lista_cedis[0] if lista_cedis else "3804"
                        
                        st.session_state.b2b_cedi = cedi_sugerido
                        st.success(f"¡Bienvenido, {credenciales[usuario][1]}!")
                        st.rerun()
                    else:
                        st.error("Usuario o contraseña incorrectos. Intenta con: 'centro' / 'centro123'")
            
            st.markdown("""
            <div style='background-color: #f1f3f5; padding: 12px; border-radius: 8px; font-size: 13px; color: #222222; text-align: center;'>
                <b>💡 Usuarios de Prueba Rápidos:</b><br>
                • centro / centro123 &nbsp;&nbsp;&nbsp;&nbsp; • pastor / pastor123<br>
                • smart / smart123 &nbsp;&nbsp;&nbsp;&nbsp; • esquina / esquina123
            </div>
            """, unsafe_allow_html=True)
    else:
        # Perfiles de lealtad de clientes basados en el historial real de la DB
        MAP_CLIENTE_PERFIL = {
            "Restaurante Centro": {"pedidos": 6, "sustituciones": 3, "tasa": 50.0},
            "Abarrotes La Esquina": {"pedidos": 7, "sustituciones": 2, "tasa": 28.57},
            "Gimnasio FitZone": {"pedidos": 8, "sustituciones": 1, "tasa": 12.50},
            "Supermercado Smart": {"pedidos": 10, "sustituciones": 0, "tasa": 0.0},
            "Taquería El Pastor": {"pedidos": 6, "sustituciones": 1, "tasa": 16.67}
        }

        # Sincronizar automáticamente el CEDI asignado según el cliente logueado
        map_cliente_cedi = {
            "Restaurante Centro": "3804",
            "Abarrotes La Esquina": "3012",
            "Gimnasio FitZone": "3803",
            "Supermercado Smart": "3003",
            "Taquería El Pastor": "3804"
        }
        cliente_nombre = st.session_state.b2b_cliente_nombre
        cedi_sugerido = map_cliente_cedi.get(cliente_nombre, "3804")
        if cedi_sugerido not in lista_cedis:
            cedi_sugerido = lista_cedis[0] if lista_cedis else "3804"
            
        if st.session_state.b2b_cedi != cedi_sugerido:
            st.session_state.b2b_cedi = cedi_sugerido
            st.session_state.b2b_pedido_procesado = False
            st.session_state.b2b_llamada_confirmada = False
            st.session_state.b2b_proteccion_activada = False
            st.rerun()

        b2b_cedi = st.session_state.b2b_cedi
        st.markdown(f"<h2 style='color: #e41e26;'>🛒 Portal de Pedidos B2B - Arca Continental</h2>", unsafe_allow_html=True)
        
        col_saludo, col_logout = st.columns([4, 1.2])
        with col_saludo:
            st.markdown(f"### 👋 ¡Hola, **{st.session_state.b2b_cliente_nombre}**!")
            st.write("Realiza tu pedido. El sistema inteligente MPC evaluará el stock del Gemelo Digital en tiempo real y tomará decisiones automatizadas de rescate de orden.")
        with col_logout:
            if st.button("🚪 Cerrar Sesión", use_container_width=True):
                st.session_state.b2b_logged_in = False
         
        # Si se acaba de finalizar la compra, mostrar pantalla de éxito
        if st.session_state.get('b2b_compra_finalizada', False):
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 25px; border-radius: 12px; border-left: 8px solid #2e7d32; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; color: #1b5e20;'>
                <h3 style='margin: 0; color: #2e7d32;'>🎉 ¡Compra finalizada!</h3>
                <p style='margin: 10px 0 0 0; font-size: 16px;'>Su pedido ha sido registrado con éxito. Se iniciará el despacho y la verificación del stock en el CEDI.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_init, _ = st.columns([1, 3])
            with col_init:
                if st.button("🛒 Realizar Nueva Compra", use_container_width=True):
                    st.session_state.b2b_compra_finalizada = False
                    st.rerun()
            st.stop()

        col_form, col_res = st.columns([1, 1.3])
        
        with col_form:
            st.subheader("📝 Agregar al Carrito")
            
            # Mostrar CEDI asignado automáticamente
            st.markdown(f"""
            <div style='background-color: #e8f5e9; padding: 12px; border-radius: 8px; border-left: 5px solid #2e7d32; margin-bottom: 12px;'>
                <span style='font-size: 11px; color: #1b5e20; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;'>🏢 CEDI de Entrega Asignado</span><br>
                <span style='font-size: 18px; font-weight: bold; color: #0f3d0f;'>CEDI {st.session_state.b2b_cedi}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Cargar productos estrella correspondientes al CEDI actual
            df_prods_cedi = get_top_productos_cedi(st.session_state.b2b_cedi)
            lista_prods = df_prods_cedi['producto'].tolist()
            
            # Formatear las opciones del selector con el precio unitario
            def format_prod_option(prod):
                precio = st.session_state.precios_productos.get(prod, 1.50)
                return f"{prod} (${precio:.2f} / caja)"
                
            b2b_producto = st.selectbox(
                "🥤 Seleccionar Producto", 
                lista_prods, 
                format_func=format_prod_option
            )
            
            # Mostrar precio unitario y calcular subtotal estimado
            precio_unitario = st.session_state.precios_productos.get(b2b_producto, 1.50)
            avg_demand_prod = int(df_prods_cedi[df_prods_cedi['producto'] == b2b_producto]['avg_qty'].iloc[0])
            
            b2b_cantidad = st.number_input("📦 Cantidad (cajas)", min_value=1, max_value=500, value=avg_demand_prod)
            subtotal_estimado = b2b_cantidad * precio_unitario
            
            st.markdown(f"""
            <div style='background-color: #f1f3f5; padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; font-size: 14px; color: #333;'>
                💵 Costo unitario: <b>${precio_unitario:.2f} USD</b><br>
                💰 Subtotal estimado: <b style='color: #e41e26; font-size: 16px;'>${subtotal_estimado:.2f} USD</b>
            </div>
            """, unsafe_allow_html=True)
            
            # Determinar stock actual y si el caso es crítico
            stock_actual = inventario.get(b2b_producto, 0)
            tipo_caso = "normal"
            if stock_actual > 0:
                if b2b_cantidad > stock_actual:
                    tipo_caso = "excede"
                elif b2b_cantidad >= 0.90 * stock_actual:
                    tipo_caso = "cercano"
            else:
                tipo_caso = "excede"
                
            # Renderizar el botón o las opciones de advertencia/autorización basadas en tipo_caso
            if tipo_caso == "normal":
                if st.button("🛒 Agregar al Carrito", use_container_width=True):
                    # Añadir al carrito
                    st.session_state.b2b_carrito.append({
                        "producto": b2b_producto,
                        "cantidad": b2b_cantidad,
                        "precio_unitario": precio_unitario,
                        "subtotal": subtotal_estimado,
                        "tipo_caso": tipo_caso,
                        "reposicion_autorizada": False,
                        "opcion_1": "-",
                        "opcion_2": "-"
                    })
                    st.toast(f"✅ {b2b_producto} agregado al carrito.")
                    st.rerun()
            else:
                # Caso crítico: cercano o excede
                if tipo_caso == "cercano":
                    st.markdown(f"""
                    <div style='background-color: #fff8e1; padding: 16px; border-radius: 8px; border-left: 6px solid #ffb300; margin-bottom: 15px; color: #5d4037; font-size: 14px; line-height: 1.5;'>
                        <h4 style='margin: 0 0 8px 0; color: #b7791f; font-weight: bold;'>⚠️ Aviso de Compromiso de Inventario (Proximidad del 10%)</h4>
                        Estimado cliente, le ofrecemos una sincera disculpa. Su pedido de <b>{b2b_cantidad} cajas</b> se encuentra muy cerca del límite del inventario disponible de <b>{b2b_producto}</b> en este CEDI (<b>{stock_actual} cajas</b>).
                        <br><br>
                        Existe la posibilidad de que alguna unidad sufra mermas menores durante el proceso de surtido en almacén. Para salvaguardar su entrega, le solicitamos autorizar la reposición por su sustituto preferido en caso de que ocurra algún incidente.
                    </div>
                    """, unsafe_allow_html=True)
                else: # excede
                    st.markdown(f"""
                    <div style='background-color: #ffe5ec; padding: 16px; border-radius: 8px; border-left: 6px solid #d81b60; margin-bottom: 15px; color: #5f1530; font-size: 14px; line-height: 1.5;'>
                        <h4 style='margin: 0 0 8px 0; color: #c2185b; font-weight: bold;'>❌ Ajuste Confirmado de Inventario</h4>
                        Estimado cliente, le ofrecemos una sincera disculpa. Su pedido de <b>{b2b_cantidad} cajas</b> excede la disponibilidad actual de <b>{b2b_producto}</b> en este CEDI (<b>{stock_actual} cajas</b>).
                        <br><br>
                        Le confirmamos que se realizará un cambio en las cajas que no podamos cubrir, reponiéndolas con el producto alternativo de su elección, y se le otorgará un descuento proporcional a la cantidad reemplazada en su factura final.
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<p style='font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #333;'>📋 Plan de Reposición Preventiva</p>", unsafe_allow_html=True)
                lista_alternativas = [p for p in inventario.keys() if p != b2b_producto]
                if not lista_alternativas:
                    lista_alternativas = ["Coca-Cola Sin Azúcar", "Coca-Cola Light"]
                    
                opcion_1 = st.selectbox("1ª Opción de Reposición (Sustituto Favorito)", lista_alternativas)
                
                lista_alternativas_2 = [p for p in lista_alternativas if p != opcion_1]
                if not lista_alternativas_2:
                    lista_alternativas_2 = ["Coca-Cola Light", "Sprite Lima Limón"]
                    
                opcion_2 = st.selectbox("2ª Opción de Reposición (Segunda Alternativa)", lista_alternativas_2)
                
                st.write("")
                
                # Botón largo de autorización dinámico según tipo de caso
                if tipo_caso == "cercano":
                    autorizar_texto = (
                        "✍️ Autorizar que, en caso de pérdida, daño o incidente durante la entrega, "
                        "se realice la reposición del producto solicitado. En dicho supuesto, se otorgará "
                        "un descuento proporcional al valor de la pérdida ocasionada. Si la entrega se "
                        "completa exitosamente y el producto llega en óptimas condiciones, no corresponderá "
                        "la aplicación de ningún descuento"
                    )
                else:
                    autorizar_texto = (
                        "✍️ Autorizo el ajuste parcial de mi pedido debido a la falta de disponibilidad "
                        "de algunos de los productos solicitados. Los productos faltantes podrán ser "
                        "reemplazados por opciones similares y recibiré un descuento proporcional por "
                        "las unidades que no se encuentran disponibles."
                    )
                if st.button(autorizar_texto, use_container_width=True):
                    # Agregar al carrito con autorización
                    st.session_state.b2b_carrito.append({
                        "producto": b2b_producto,
                        "cantidad": b2b_cantidad,
                        "precio_unitario": precio_unitario,
                        "subtotal": subtotal_estimado,
                        "tipo_caso": tipo_caso,
                        "reposicion_autorizada": True,
                        "opcion_1": opcion_1,
                        "opcion_2": opcion_2
                    })
                    # Agregar aviso a la bandeja de entrada del operador
                    if 'b2b_alertas_operador' in st.session_state:
                        st.session_state.b2b_alertas_operador.insert(0, {
                            "cliente": st.session_state.b2b_cliente_nombre,
                            "producto": b2b_producto,
                            "cantidad": b2b_cantidad,
                            "stock_actual": stock_actual,
                            "tipo_caso": tipo_caso,
                            "fecha": time.strftime("%H:%M:%S"),
                            "leido": False
                        })
                    st.toast(f"✅ {b2b_producto} (con reposición) agregado al carrito.")
                    st.rerun()

        with col_res:
            st.subheader("🛒 Carrito de Compras")
            
            if not st.session_state.b2b_carrito:
                st.info("El carrito está vacío. Agrega productos usando el formulario de la izquierda.")
            else:
                # Mostrar tabla de productos agregados
                items = []
                for idx, item in enumerate(st.session_state.b2b_carrito):
                    rep_str = "Normal (Seguro)"
                    if item["reposicion_autorizada"]:
                        rep_str = f"Sustituir por: {item['opcion_1']}"
                    items.append({
                        "Producto": item["producto"],
                        "Cant.": item["cantidad"],
                        "P. Unit.": f"${item['precio_unitario']:.2f}",
                        "Subtotal": f"${item['subtotal']:.2f}",
                        "Estatus Stock": rep_str
                    })
                
                df_carrito = pd.DataFrame(items)
                st.dataframe(df_carrito, use_container_width=True, hide_index=True)
                
                # Calcular total
                grand_total = sum(item["subtotal"] for item in st.session_state.b2b_carrito)
                
                st.markdown(f"""
                <div style='text-align: right; font-size: 20px; font-weight: bold; margin-top: 10px; margin-bottom: 20px; color: #333;'>
                    Total a Pagar: <span style='color: #e41e26;'>${grand_total:.2f} USD</span>
                </div>
                """, unsafe_allow_html=True)
                
                col_pagar, col_vaciar = st.columns([2, 1])
                with col_pagar:
                    if st.button("💳 Pagar Pedido", type="primary", use_container_width=True):
                        # Limpiar carrito y marcar como finalizado
                        st.session_state.b2b_carrito = []
                        st.session_state.b2b_compra_finalizada = True
                        st.rerun()
                with col_vaciar:
                    if st.button("🗑️ Vaciar Carrito", use_container_width=True):
                        st.session_state.b2b_carrito = []
                        st.rerun()
        st.stop()

# ----------------- TABS PRINCIPALES -----------------
tab1, tab2, tab3 = st.tabs([
    "📥 Bandeja de Entrada del Operador", 
    "📊 Dashboard CEDI y Analíticas",
    "🚀 Smart Order Rescue"
])

# ----------------- TAB 1: BANDEJA DE ENTRADA DEL OPERADOR -----------------
with tab1:
    st.header(f"📥 Bandeja de Entrada del Operador - CEDI {cedi_seleccionado}")
    st.write("Gestiona las alertas operativas de stock en tiempo real y revisa los perfiles de lealtad de tus clientes:")

    # Dividir en dos columnas para una vista premium
    col_container = st.container()
    col_menu, col_detail = col_container.columns([1, 1.8])

    with col_menu:
        # Selección de carpeta/bandeja
        bandeja = st.radio(
            "📁 Seleccionar Bandeja", 
            ["🚨 Alertas de Stock Crítico", "👥 Historial de Clientes & Lealtad"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if bandeja == "🚨 Alertas de Stock Crítico":
            st.subheader("Notificaciones de Riesgo")
            alertas = st.session_state.get('b2b_alertas_operador', [])
            
            if not alertas:
                st.success("✅ No hay alertas de stock pendientes en la red.")
                selected_alert_idx = None
            else:
                # Mostrar botones tipo lista de correos
                selected_alert_idx = 0
                for idx, alert in enumerate(alertas):
                    severity = "🔴 CRÍTICO" if alert["tipo_caso"] == "excede" else "🟠 AVISO"
                    # Resaltar si no está leído
                    unread_prefix = "✉️ " if not alert["leido"] else "📖 "
                    btn_label = f"{unread_prefix} [{alert['fecha']}] {alert['cliente']} - {severity}"
                    
                    if st.button(btn_label, key=f"alert_btn_{idx}", use_container_width=True):
                        # Marcar como leída
                        st.session_state.b2b_alertas_operador[idx]["leido"] = True
                        st.session_state.selected_alert_index = idx
                        st.rerun()
                
                selected_alert_idx = st.session_state.get('selected_alert_index', 0)
                if selected_alert_idx >= len(alertas):
                    selected_alert_idx = 0
                    
        else: # Historial de Clientes & Lealtad
            st.subheader("Confirmaciones de Descuento")
            st.write("Solicitudes pendientes basadas en incidencias de stock:")
            clientes_lista = ["Restaurante Centro", "Abarrotes La Esquina", "Gimnasio FitZone", "Supermercado Smart", "Taquería El Pastor"]
            
            MAP_CLIENTE_RESUMEN = {
                "Restaurante Centro": "50.0% mermas (Crítico)",
                "Abarrotes La Esquina": "28.6% mermas (Regular)",
                "Gimnasio FitZone": "12.5% mermas (Estable)",
                "Supermercado Smart": "0.0% mermas (Excelente)",
                "Taquería El Pastor": "16.7% mermas (Estable)"
            }
            
            for c_name in clientes_lista:
                confirmed_status = "✅ [CONFIRMADO]" if st.session_state.get('descuentos_confirmados', {}).get(c_name, False) else "📩 [PENDIENTE]"
                resumen = MAP_CLIENTE_RESUMEN.get(c_name, "")
                btn_label = f"{confirmed_status} {c_name} ({resumen})"
                
                if st.button(btn_label, key=f"client_btn_{c_name}", use_container_width=True):
                    st.session_state.selected_client_name = c_name
                    st.rerun()
            
            selected_client = st.session_state.get('selected_client_name', "Restaurante Centro")

    with col_detail:
        if bandeja == "🚨 Alertas de Stock Crítico":
            st.subheader("🔎 Detalle de Alerta Operativa")
            
            alertas = st.session_state.get('b2b_alertas_operador', [])
            if alertas and selected_alert_idx is not None:
                alert = alertas[selected_alert_idx]
                
                # Diseño premium de la alerta seleccionada
                bg_color = "#ffe5ec" if alert["tipo_caso"] == "excede" else "#fff8e1"
                border_color = "#d81b60" if alert["tipo_caso"] == "excede" else "#ffb300"
                text_color = "#5f1530" if alert["tipo_caso"] == "excede" else "#5d4037"
                
                st.markdown(f"""
                <div style='background-color: {bg_color}; padding: 18px; border-radius: 10px; border-left: 6px solid {border_color}; margin-bottom: 20px; color: {text_color};'>
                    <h3 style='margin: 0 0 10px 0; color: {border_color}; font-weight: bold;'>
                        { '🚨 Stock Excedido (Faltante Confirmado)' if alert['tipo_caso'] == 'excede' else '⚠️ Stock Cercano (Riesgo del 10%)' }
                    </h3>
                    <b>Cliente:</b> {alert['cliente']}<br>
                    <b>Producto Solicitado:</b> {alert['producto']}<br>
                    <b>Cantidad en Pedido:</b> {alert['cantidad']} cajas<br>
                    <b>Inventario Disponible:</b> {alert['stock_actual']} cajas<br>
                    <b>Déficit/Diferencia:</b> {max(0, alert['cantidad'] - alert['stock_actual'])} cajas<br>
                    <b>Hora de Registro:</b> {alert['fecha']}
                </div>
                """, unsafe_allow_html=True)
                

            else:
                st.info("Selecciona una notificación de la lista para ver su detalle y tomar cartas en el asunto.")
                
        else: # Historial de Clientes & Lealtad
            st.subheader(f"🔎 Confirmación de Descuento: {selected_client}")
            
            # Datos de perfil reales mapeados
            MAP_CLIENTE_PERFIL_REAL = {
                "Restaurante Centro": {"pedidos": 6, "sustituciones": 3, "tasa": 50.0, "calif": "Riesgo de Abandono ⚠️", "color": "red", "prom_cajas": 25, "problema": "Sufre alta tasa de mermas e incidencias críticas de stock en almacén. Requiere compensación prioritaria."},
                "Abarrotes La Esquina": {"pedidos": 7, "sustituciones": 2, "tasa": 28.57, "calif": "Regular 👤", "color": "orange", "prom_cajas": 15, "problema": "Presenta desabastos esporádicos en pedidos de volumen medio."},
                "Gimnasio FitZone": {"pedidos": 8, "sustituciones": 1, "tasa": 12.50, "calif": "Buen Cliente ⭐", "color": "green", "prom_cajas": 12, "problema": "Buen historial. Mantiene incidencias mínimas de stock."},
                "Supermercado Smart": {"pedidos": 10, "sustituciones": 0, "tasa": 0.0, "calif": "Cliente Excelente 🏆", "color": "blue", "prom_cajas": 45, "problema": "Excelente historial de pedidos. Sin problemas reportados recientemente."},
                "Taquería El Pastor": {"pedidos": 6, "sustituciones": 1, "tasa": 16.67, "calif": "Buen Cliente ⭐", "color": "green", "prom_cajas": 18, "problema": "Riesgo moderado de desabasto en pedidos pico de fin de semana."}
            }
            
            profile = MAP_CLIENTE_PERFIL_REAL.get(selected_client, {"pedidos": 5, "sustituciones": 0, "tasa": 0.0, "calif": "Regular", "color": "grey", "prom_cajas": 10, "problema": "Sin incidentes graves."})
            
            # Mostrar KPIs del cliente
            kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
            with kpi_c1:
                st.metric("Total Pedidos", profile["pedidos"])
            with kpi_c2:
                st.metric("Sustituciones Recibidas", profile["sustituciones"])
            with kpi_c3:
                st.metric("Tasa de Sustitución", f"{profile['tasa']:.1f}%")
                
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid {profile["color"]}; margin-bottom: 20px; color: #333;'>
                <b>Clasificación de Cliente:</b> <span style='color: {profile["color"]}; font-weight: bold;'>{profile["calif"]}</span><br>
                <b>Tamaño Promedio de Pedido:</b> {profile["prom_cajas"]} cajas/pedido<br>
                <b>Incidencia Actual / Historial de Problema:</b> {profile["problema"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Descuento sugerido en base a pedidos, lealtad y tamaño de orden (Capped a 5%)
            tasa_exito = 100.0 - profile["tasa"]
            descuento_sugerido_calculado = min(5.0, (profile["pedidos"] * 0.25) + (profile["prom_cajas"] * 0.04) + (tasa_exito * 0.01))
            descuento_sugerido_calculado = round(descuento_sugerido_calculado, 1)
            
            st.subheader("🎟️ Evaluación y Confirmación de Descuento")
            st.write("El sistema calcula un descuento de lealtad sugerido para mitigar el impacto del problema presentado:")
            
            st.info(f"💡 **Descuento Recomendado:** **{descuento_sugerido_calculado}%** (Capped al 5%)")
            
            # Obtener descuento manual actual de sesión o usar el sugerido por defecto
            desc_actual = st.session_state.descuentos_manuales.get(selected_client, descuento_sugerido_calculado)
            
            # Input para que el operador modifique/altere el descuento en tiempo real
            nuevo_desc = st.slider(
                f"Ajustar Descuento de Compensación para {selected_client} (%)", 
                min_value=0.0, 
                max_value=5.0, 
                value=float(desc_actual), 
                step=0.5,
                key=f"slider_desc_{selected_client}"
            )
            
            # Guardar en estado de sesión el descuento modificado
            st.session_state.descuentos_manuales[selected_client] = nuevo_desc
            
            # Botón para confirmar y aplicar descuento
            st.write("")
            col_confirm, col_status = st.columns([1, 1])
            with col_confirm:
                if st.button("✅ Confirmar y Aplicar Descuento", use_container_width=True):
                    st.session_state.descuentos_confirmados[selected_client] = True
                    st.rerun()
            
            with col_status:
                if st.session_state.descuentos_confirmados.get(selected_client, False):
                    st.success(f"🎉 ¡Descuento de **{nuevo_desc}%** Aplicado con éxito!")
                else:
                    st.warning("⏳ Pendiente de confirmación por el operador.")

    # ----------------- PANEL DE PRIORIZACIÓN A ANCHO COMPLETO -----------------
    if bandeja == "🚨 Alertas de Stock Crítico":
        alertas = st.session_state.get('b2b_alertas_operador', [])
        if alertas:
            st.markdown("---")
            st.subheader("📊 Panel de Priorización de Abasto CEDI")
            
            col_priority, col_chart = st.columns([1.1, 1.3])
            
            # Generar datos de prioridad
            prioridades = []
            for a in alertas:
                dif = max(0, a["cantidad"] - a["stock_actual"])
                prio_score = (dif / a["stock_actual"]) * 100 if a["stock_actual"] > 0 else 100.0
                if a["tipo_caso"] == "cercano":
                    prio_score = 15.0 # Prioridad baja para alertas de cercanía
                prioridades.append({
                    "Alerta": f"{a['cliente']} ({a['producto']})",
                    "Severidad (%)": round(prio_score, 1)
                })
            
            # Ordenar prioridades de mayor a menor severidad
            prioridades = sorted(prioridades, key=lambda x: x["Severidad (%)"], reverse=True)
            lista_nombres_alertas_ordenados = [p["Alerta"] for p in prioridades]
            
            with col_priority:
                st.markdown("🎯 **Prioridades de Abasto Detectadas:**")
                st.write("Las alertas están ordenadas automáticamente de mayor a menor gravedad según el déficit:")
                
                alerta_seleccionada_prio = st.selectbox(
                    "Seleccionar CEDI / Pedido Crítico:",
                    lista_nombres_alertas_ordenados if lista_nombres_alertas_ordenados else ["No hay alertas"],
                    key="prio_selectbox_fullwidth"
                )
                
                # Buscar el score de severidad
                score_prio = 0.0
                for p in prioridades:
                    if p["Alerta"] == alerta_seleccionada_prio:
                        score_prio = p["Severidad (%)"]
                        break
                
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 6px; margin-bottom: 12px; border: 1px solid #ddd; color: #333;'>
                    📢 <b>Prioridad del Sistema:</b> { '🔴 Crítica/Urgente' if score_prio >= 50.0 else ('🟠 Alta' if score_prio >= 30.0 else '🟡 Media') }<br>
                    📈 <b>Severidad de Desabasto:</b> {score_prio}%
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Mandar a Pedir Stock", use_container_width=True, key="prio_btn_fullwidth"):
                    st.success(f"✅ Se ha enviado una orden de reabastecimiento urgente para: **{alerta_seleccionada_prio}**.")
                    
            with col_chart:
                st.markdown("📈 **Gráfica de Severidad:**")
                df_prio = pd.DataFrame(prioridades)
                if not df_prio.empty:
                    st.bar_chart(data=df_prio.set_index("Alerta")["Severidad (%)"], color="#e41e26")

# ----------------- TAB 2: DASHBOARD CEDI Y ANALITICAS -----------------
with tab2:
    st.header(f"Métricas y Analíticas del CEDI {cedi_seleccionado}")
    st.write(f"Estadísticas específicas para el CEDI seleccionado en comparación con el promedio general de la red:")
    
    conn = get_db_connection()
    
    total_orders_cedi = pd.read_sql_query("SELECT count(distinct id_pedido) as c FROM orders WHERE cedis = ?;", conn, params=(cedi_seleccionado,)).iloc[0]['c']
    sub_orders_cedi = pd.read_sql_query("""
        SELECT count(distinct s.id_pedido) as c 
        FROM sustituciones s
        JOIN orders o ON s.id_pedido = o.id_pedido
        WHERE o.cedis = ?;
    """, conn, params=(cedi_seleccionado,)).iloc[0]['c']
    
    sub_lines_cedi = pd.read_sql_query("""
        SELECT count(s.id_linea) as c 
        FROM sustituciones s
        JOIN orders o ON s.id_pedido = o.id_pedido
        WHERE o.cedis = ?;
    """, conn, params=(cedi_seleccionado,)).iloc[0]['c']
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Pedidos CEDI {cedi_seleccionado}</div>
            <div class='metric-value'>{total_orders_cedi:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        tasa_sub = (sub_orders_cedi / total_orders_cedi * 100) if total_orders_cedi > 0 else 0.0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Tasa de Sustitución del CEDI</div>
            <div class='metric-value'>{tasa_sub:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Líneas Sustituidas en CEDI</div>
            <div class='metric-value'>{sub_lines_cedi:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader(f"🏆 Productos Más Sustituidos en CEDI {cedi_seleccionado}")
        df_top_prod = pd.read_sql_query("""
            SELECT s.nombre_sku_solicitado as Producto, count(*) as Sustituciones
            FROM sustituciones s
            JOIN orders o ON s.id_pedido = o.id_pedido
            WHERE o.cedis = ?
            GROUP BY s.nombre_sku_solicitado
            ORDER BY Sustituciones DESC
            LIMIT 5;
        """, conn, params=(cedi_seleccionado,))
        st.dataframe(df_top_prod, use_container_width=True)
        
    with col_g2:
        st.subheader("📍 Los 5 CEDIS con más problemas en la red")
        df_top_cedis = pd.read_sql_query("""
            SELECT o.cedis as CEDI, count(s.id_linea) as Sustituciones
            FROM orders o
            JOIN sustituciones s ON o.id_pedido = s.id_pedido
            GROUP BY o.cedis
            ORDER BY Sustituciones DESC
            LIMIT 5;
        """, conn)
        st.dataframe(df_top_cedis, use_container_width=True)
        
    st.subheader(f"🔄 Parejas de Cambio Más Frecuentes en CEDI {cedi_seleccionado}")
    df_pairs = pd.read_sql_query("""
        SELECT 
            s.nombre_sku_solicitado as "Original Solicitado",
            s.nombre_sku_solicitado_cambio as "Sustituto Entregado",
            count(*) as "Cantidad de Veces"
        FROM sustituciones s
        JOIN orders o ON s.id_pedido = o.id_pedido
        WHERE o.cedis = ?
        GROUP BY s.nombre_sku_solicitado, s.nombre_sku_solicitado_cambio
        ORDER BY "Cantidad de Veces" DESC
        LIMIT 5;
    """, conn, params=(cedi_seleccionado,))
    st.dataframe(df_pairs, use_container_width=True)
    
    conn.close()

# ----------------- TAB 3: SMART ORDER RESCUE -----------------
with tab3:
    render_smart_order_rescue(cedi_seleccionado)

# ----------------- ASISTENTE VIRTUAL FLOTANTE (GEMINI CHATBOT) -----------------
# CSS para posicionar el botón y la ventana de chat de forma fija en la esquina inferior derecha
st.markdown("""
<style>
    /* Estilos para el contenedor flotante del botón */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="element-container"] .my-marker-chat-btn) {
        position: fixed !important;
        bottom: 20px !important;
        right: 20px !important;
        z-index: 999999 !important;
        width: auto !important;
        background-color: transparent !important;
    }
    
    /* Estilos para el botón flotante en sí */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="element-container"] .my-marker-chat-btn) button {
        border-radius: 50% !important;
        width: 60px !important;
        height: 60px !important;
        background-color: #e41e26 !important;
        color: white !important;
        border: none !important;
        font-size: 26px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: transform 0.2s ease !important;
    }
    
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="element-container"] .my-marker-chat-btn) button:hover {
        transform: scale(1.08) !important;
        background-color: #c31820 !important;
    }
    
    /* Estilos para el contenedor flotante de la ventana de chat */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="element-container"] .my-marker-chat-window) {
        position: fixed !important;
        bottom: 90px !important;
        right: 20px !important;
        width: 360px !important;
        height: 520px !important;
        background-color: white !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.18) !important;
        border: 1px solid #ddd !important;
        z-index: 999999 !important;
        padding: 15px !important;
        display: flex !important;
        flex-direction: column !important;
        overflow-y: hidden !important;
    }
    
    /* Quitar padding innecesario de streamlit en el bloque del chat */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="element-container"] .my-marker-chat-window) > div {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# 1. Ventana de chat flotante (solo si está abierta)
if st.session_state.chat_flotante_abierto:
    chat_window_container = st.container()
    with chat_window_container:
        st.markdown('<div class="my-marker-chat-window"></div>', unsafe_allow_html=True)
        st.markdown("<h4 style='margin: 0 0 10px 0; color:#e41e26; font-family:sans-serif;'>🤖 Asistente Virtual AC</h4>", unsafe_allow_html=True)
        
        # API Key de Gemini (usando el fallback ingresado o variables de entorno)
        api_key_env = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "") or st.session_state.gemini_api_key_temp
        
        if not api_key_env:
            st.info("💡 Por favor, configura una API Key de Gemini.")
        else:
            # Contenedor scrollable para mensajes
            chat_box = st.container(height=330)
            with chat_box:
                if len(st.session_state.chatbot_historial) == 0:
                    st.write("¡Hola! Pregúntame sobre el funcionamiento del portal, los usuarios o los modelos predictivos del Gemelo Digital.")
                    st.markdown("<p style='font-size:12px; font-weight:bold; margin-top:10px; margin-bottom:5px; color:#555;'>Preguntas frecuentes:</p>", unsafe_allow_html=True)
                    preguntas_sug = [
                        "¿Cómo funciona el portal B2B?",
                        "¿Qué hace el Gemelo Digital?",
                        "¿Cómo predice XGBoost?",
                        "¿Qué es Smart Order Rescue?"
                    ]
                    for q in preguntas_sug:
                        if st.button(q, key=f"sug_flotante_{q}", use_container_width=True):
                            st.session_state.chatbot_historial.append({"role": "user", "content": q})
                            respuesta = llamar_api_gemini(q, api_key_env)
                            st.session_state.chatbot_historial.append({"role": "assistant", "content": respuesta})
                            st.rerun()
                else:
                    for msg in st.session_state.chatbot_historial:
                        role_css = "chat-agent" if msg["role"] == "assistant" else "chat-user"
                        emoji = "🤖" if msg["role"] == "assistant" else "👤"
                        st.markdown(f"""
                        <div class="chat-bubble {role_css}" style="max-width: 90%; font-size: 13px; margin-bottom: 8px;">
                            <b>{emoji} { 'Asistente' if msg['role'] == 'assistant' else 'Tú' }:</b><br>
                            {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Entrada de texto (chat_input)
            user_input = st.chat_input("Escribe tu duda...", key="chatbot_flotante_input_field")
            if user_input:
                st.session_state.chatbot_historial.append({"role": "user", "content": user_input})
                respuesta = llamar_api_gemini(user_input, api_key_env)
                st.session_state.chatbot_historial.append({"role": "assistant", "content": respuesta})
                st.rerun()
        
        # Botón para vaciar chat
        if len(st.session_state.chatbot_historial) > 0:
            if st.button("🗑️ Limpiar Conversación", use_container_width=True, key="clear_chat_flotante_btn"):
                st.session_state.chatbot_historial = []
                st.rerun()

# 2. Botón flotante para abrir/cerrar chat
chat_btn_container = st.container()
with chat_btn_container:
    st.markdown('<div class="my-marker-chat-btn"></div>', unsafe_allow_html=True)
    # Botón flotante con ícono de chat o cierre
    btn_label = "✖" if st.session_state.chat_flotante_abierto else "💬"
    if st.button(btn_label, key="btn_chat_flotante", help="Abrir/Cerrar Asistente Inteligente AC"):
        st.session_state.chat_flotante_abierto = not st.session_state.chat_flotante_abierto
        st.rerun()



