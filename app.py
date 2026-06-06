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

# Importamos la función de optimización bayesiana
from optimizacion_bayesiana import ejecutar_optimizacion_bayesiana

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

# Inicialización de estados de comprador B2B
if 'b2b_logged_in' not in st.session_state:
    st.session_state.b2b_logged_in = False
    st.session_state.b2b_user = ""
    st.session_state.b2b_cliente_nombre = ""
    
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

# Inputs de APIs e Integraciones
st.sidebar.markdown("<h2 style='color: #e41e26;'>🔑 APIs Configuración</h2>", unsafe_allow_html=True)
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key (Para audio de texto)", type="password")
elevenlabs_agent_id = st.sidebar.text_input("ElevenLabs Agent ID (Para hablar con micrófono)", value="")

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
                st.session_state.b2b_user = ""
                st.session_state.b2b_cliente_nombre = ""
                st.session_state.b2b_pedido_procesado = False
                st.session_state.b2b_llamada_confirmada = False
                st.session_state.b2b_proteccion_activada = False
                st.rerun()
                
        st.markdown("---")
        
        col_form, col_res = st.columns([1, 1.5])
        
        with col_form:
            st.subheader("📝 Formulario de Pedido")
            
            # Mostrar CEDI asignado automáticamente
            st.markdown(f"""
            <div style='background-color: #e8f5e9; padding: 12px; border-radius: 8px; border-left: 5px solid #2e7d32; margin-bottom: 12px;'>
                <span style='font-size: 11px; color: #1b5e20; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;'>🏢 CEDI de Entrega Asignado (El más cercano)</span><br>
                <span style='font-size: 18px; font-weight: bold; color: #0f3d0f;'>CEDI {st.session_state.b2b_cedi}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # --- FICHA DE LEALTAD ---
            perfil = MAP_CLIENTE_PERFIL.get(st.session_state.b2b_cliente_nombre, {"pedidos": 5, "sustituciones": 0, "tasa": 0.0})
            tasa_sub = perfil["tasa"]
            
            if tasa_sub >= 35.0:
                semaforo_color = "#d32f2f" # rojo
                semaforo_text = "🚨 Tolerancia Crítica (Riesgo de Abandono)"
                semaforo_desc = "El cliente ha sufrido múltiples sustituciones históricas. El MPC priorizará entregar el producto original."
            elif tasa_sub >= 15.0:
                semaforo_color = "#f57c00" # naranja
                semaforo_text = "⚠️ Tolerancia Moderada"
                semaforo_desc = "El cliente ha tenido sustituciones previas de forma intermitente."
            else:
                semaforo_color = "#388e3c" # verde
                semaforo_text = "✅ Tolerancia Alta (Cliente Satisfecho)"
                semaforo_desc = "Historial limpio. Tolerancia adecuada para ofrecer sustituciones con descuento."
                
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-top: 4px solid {semaforo_color}; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.03); color: #333333;'>
                <h4 style='margin: 0 0 5px 0; color: {semaforo_color}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: bold;'>🛡️ Ficha de Lealtad del Cliente</h4>
                <p style='margin: 0; font-size: 12px; color: #222222;'>Pedidos Totales: <b>{perfil["pedidos"]}</b> | Cambios: <b>{perfil["sustituciones"]} ({tasa_sub:.1f}%)</b></p>
                <p style='margin: 4px 0 0 0; font-size: 11px; font-weight: bold; color: {semaforo_color};'>{semaforo_text}</p>
                <p style='margin: 3px 0 0 0; font-size: 10px; color: #444444; line-height: 1.2;'>{semaforo_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cargar productos estrella correspondientes al CEDI actual
            df_prods_cedi = get_top_productos_cedi(st.session_state.b2b_cedi)
            lista_prods = df_prods_cedi['producto'].tolist()
            
            b2b_producto = st.selectbox("🥤 Seleccionar Producto Solicitado", lista_prods)
            
            avg_demand_prod = int(df_prods_cedi[df_prods_cedi['producto'] == b2b_producto]['avg_qty'].iloc[0])
            b2b_cantidad = st.number_input("📦 Cantidad (cajas)", min_value=1, max_value=500, value=avg_demand_prod)
            
            enviar_pedido = st.button("🚀 Enviar Pedido", use_container_width=True)
            
            if enviar_pedido:
                if model is None:
                    st.error("Error: El modelo de Machine Learning no ha sido entrenado. Corre 'entrenar_modelo.py' primero.")
                else:
                    # 1. Capa ML: Predicción de desabasto
                    input_data = pd.DataFrame([{
                        'nombre_solicitado': b2b_producto,
                        'quantity': b2b_cantidad
                    }])
                    prob_desabasto = model.predict_proba(input_data)[0][1]
                    
                    # Stock actual en Gemelo Digital
                    stock_actual = inventario.get(b2b_producto, 0)
                    if stock_actual >= b2b_cantidad:
                        prob_real = max(0.01, prob_desabasto * 0.1)
                    else:
                        deficit = b2b_cantidad - stock_actual
                        prob_real = min(0.99, prob_desabasto * 1.5 + (deficit / b2b_cantidad))
                    
                    # 2. Capa MPC: Decisiones Operativas Automatizadas (Costos Fijos)
                    otros_prods = [p for p in inventario.keys() if p != b2b_producto]
                    b2b_sustituto_sugerido = otros_prods[0] if otros_prods else "Coca-Cola Sin Azúcar"
                    
                    opciones = {
                        "A) No hacer nada (Permitir Sustitución en CEDI)": {
                            "costo_fijo": 0.0,
                            "prob_falla": prob_real,
                            "desc": "El pedido sigue su curso normal. Existe un riesgo alto de queja o que no te llegue lo solicitado."
                        },
                        "B) Reubicación de stock urgente (Mover de CEDI cercano)": {
                            "costo_fijo": 35.0,
                            "prob_falla": 0.05,
                            "desc": "Se activó la reubicación de inventario urgente desde un CEDI vecino. Tu entrega está garantizada sin cargos extra."
                        },
                        "C) Notificar al cliente y autorizar sustituto con descuento": {
                            "costo_fijo": 8.0,
                            "prob_falla": prob_real * 0.1,
                            "desc": f"Se sugiere cambio por '{b2b_sustituto_sugerido}' con descuento."
                        }
                    }
                    
                    # Calcular decisión sin protección de lealtad (para saber si influyó en el cambio)
                    nombres_sin = []
                    costos_sin = []
                    for nombre, opt in opciones.items():
                        costo_sin = opt["costo_fijo"] + (opt["prob_falla"] * 120.0)
                        nombres_sin.append(nombre)
                        costos_sin.append(costo_sin)
                    df_sin = pd.DataFrame({"Acción": nombres_sin, "Costo": costos_sin}).sort_values(by="Costo")
                    mejor_accion_sin = df_sin.iloc[0]["Acción"]
                    
                    # Calcular decisión con protección de lealtad
                    tasa_historica = perfil.get("tasa", 0.0)
                    penalizacion_lealtad = (tasa_historica / 100.0) * 80.0  # hasta $40 USD extra por insatisfacción
                    
                    nombres = []
                    costos = []
                    descripciones = []
                    for nombre, opt in opciones.items():
                        costo_esperado = opt["costo_fijo"] + (opt["prob_falla"] * 120.0)
                        # Sumar penalización por lealtad a las opciones que NO entregan el producto original (A y C)
                        if "A)" in nombre or "C)" in nombre:
                            costo_esperado += penalizacion_lealtad
                        nombres.append(nombre)
                        costos.append(costo_esperado)
                        descripciones.append(opt["desc"])
                        
                    df_mpc = pd.DataFrame({
                        "Acción Propuesta": nombres,
                        "Costo Esperado ($ USD)": costos,
                        "Descripción": descripciones
                    }).sort_values(by="Costo Esperado ($ USD)")
                    
                    mejor_accion = df_mpc.iloc[0]["Acción Propuesta"]
                    menor_costo = df_mpc.iloc[0]["Costo Esperado ($ USD)"]
                    
                    # Si la decisión cambió debido a la penalización de lealtad hacia la Reubicación (B)
                    proteccion_activada = False
                    if mejor_accion != mejor_accion_sin and "B)" in mejor_accion:
                        proteccion_activada = True
                    
                    st.session_state.b2b_pedido_procesado = True
                    st.session_state.b2b_producto = b2b_producto
                    st.session_state.b2b_cantidad = b2b_cantidad
                    st.session_state.b2b_prob_real = prob_real
                    st.session_state.b2b_mejor_accion = mejor_accion
                    st.session_state.b2b_menor_costo = menor_costo
                    st.session_state.b2b_df_mpc = df_mpc
                    st.session_state.b2b_sustituto_sugerido = b2b_sustituto_sugerido
                    st.session_state.b2b_simular_llamada_clic = False
                    st.session_state.b2b_llamada_confirmada = False
                    st.session_state.b2b_proteccion_activada = proteccion_activada
                    st.rerun()

        with col_res:
            if st.session_state.b2b_pedido_procesado:
                st.subheader("🤖 Estatus del Pedido (Decisión MPC Automática)")
                
                mejor_accion = st.session_state.b2b_mejor_accion
                
                if st.session_state.get('b2b_proteccion_activada', False):
                    cliente_nombre = st.session_state.b2b_cliente_nombre
                    # Obtenemos la tasa de la ficha mapeada para mostrarla en el banner
                    tasa_sub = MAP_CLIENTE_PERFIL.get(cliente_nombre, {}).get("tasa", 0.0)
                    st.markdown(f"""
                    <div style='background-color: #fff8e1; padding: 15px; border-radius: 8px; border-left: 6px solid #ffb300; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); color: #5d4037;'>
                        <h4 style='margin: 0; color: #5d4037; font-size: 14px; font-weight: bold;'>🛡️ Regla de Protección de Lealtad Activada</h4>
                        <p style='margin: 5px 0 0 0; font-size: 13px; color: #5d4037; line-height: 1.4;'>
                            El sistema detectó que este cliente tiene una tasa de sustitución histórica crítica (<b>{tasa_sub:.1f}%</b>). 
                            Para proteger su relación con Arca Continental y evitar su descontento, el MPC penalizó las opciones de cambio de producto y 
                            eligió automáticamente <b>entregarle su producto original</b> (mediante la acción <b>{mejor_accion}</b>).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                producto = st.session_state.b2b_producto
                cantidad = st.session_state.b2b_cantidad
                sustituto_sugerido = st.session_state.b2b_sustituto_sugerido
                prob_real = st.session_state.b2b_prob_real
                stock_actual = inventario.get(producto, 0)
                
                if st.session_state.b2b_llamada_confirmada:
                    st.success("✅ PEDIDO CONFIRMADO CON SUSTITUCIÓN ACEPTADA")
                    st.markdown(f"""
                    **Resumen de la Orden Corregida:**
                    - Producto Original: ~*{producto}*~ *(Sin Stock)*
                    - Producto Entregado: **{sustituto_sugerido}**
                    - Cantidad: **{cantidad} cajas**
                    - Descuento Especial Aplicado: **$8.00 USD**
                    - Estatus de Despacho: *En preparación para salida*
                    """)
                else:
                    if "A)" in mejor_accion:
                        st.info("📝 PEDIDO REGISTRADO Y EN CURSO")
                        st.markdown(f"""
                        Tu pedido de **{cantidad} cajas de {producto}** ha sido ingresado al sistema.
                        
                        *Nota del Gemelo Digital:* Se estima un riesgo de desabasto muy bajo ({prob_real*100:.1f}%), por lo que tu pedido sigue su ruta convencional.
                        """)
                    elif "B)" in mejor_accion:
                        st.success("🚚 PEDIDO APROBADO (LOGÍSTICA PREDICTIVA ACTIVADA)")
                        st.markdown(f"""
                        Tu pedido de **{cantidad} cajas de {producto}** ha sido aprobado con entrega garantizada.
                        
                        *Acción Automatizada:* Detectamos un riesgo moderado de falta de stock en este CEDI ({prob_real*100:.1f}%). El sistema MPC de Arca Continental ha ordenado la **reubicación de inventario urgente** desde un CEDI vecino para cubrir tu orden a tiempo y sin costo adicional.
                        """)
                    elif "C)" in mejor_accion:
                        st.error("⚠️ AJUSTE DE PEDIDO REQUERIDO (RECOMENDACIÓN MPC)")
                        st.markdown(f"""
                        Detectamos un desabasto crítico de **{producto}** en el CEDI {b2b_cedi}. 
                        
                        El sistema MPC recomienda la opción **C: Ofrecer sustituto '{sustituto_sugerido}' con descuento**. 
                        Por favor, atiende la llamada del agente virtual de soporte o realiza la simulación a continuación:
                        """)
                        
                        if elevenlabs_agent_id:
                            st.write("🤖 **Llamada Telefónica con Agente de Voz IA en Vivo:**")
                            
                            order_id_esc = f"AC-{b2b_cedi}-1042".replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
                            customer_name_esc = str(st.session_state.b2b_cliente_nombre).replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
                            product_name_esc = str(producto).replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
                            quantity_esc = str(cantidad).replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
                            substitute_name_esc = str(sustituto_sugerido).replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
                            discount_value_esc = "8.0 USD"
                            
                            components.html(f"""
                            <div style="display: flex; justify-content: center; align-items: center; height: 180px; flex-direction: column; font-family: sans-serif;">
                                <elevenlabs-convai 
                                    id="el-agent-b2b"
                                    agent-id="{elevenlabs_agent_id}"
                                ></elevenlabs-convai>
                                <script>
                                    const el = document.getElementById("el-agent-b2b");
                                    const vars = {{
                                        "order_id": "{order_id_esc}",
                                        "customer_name": "{customer_name_esc}",
                                        "product_name": "{product_name_esc}",
                                        "quantity": "{quantity_esc}",
                                        "substitute_name": "{substitute_name_esc}",
                                        "discount_value": "{discount_value_esc}"
                                    }};
                                    el.setAttribute("dynamic-variables", JSON.stringify(vars));
                                    console.log("B2B ElevenLabs Dynamic Variables:", vars);
                                </script>
                                <script src="https://elevenlabs.io/convai-widget/index.js" type="text/javascript"></script>
                                <p style="margin-top: 10px; font-size: 12px; color: #666;">Usa tu micrófono para hablar con el agente flotante de Arca Continental.</p>
                            </div>
                            """, height=220)
                        else:
                            st.write("El sistema generará el audio de la llamada vía API. Si quieres hablar por micrófono, ingresa tu `Agent ID` de ElevenLabs en la barra lateral.")
                            
                            if st.button("📞 Simular Llamada de Voz con IA (API)", use_container_width=True, key="b2b_sim_ll"):
                                st.session_state.b2b_simular_llamada_clic = True
                            
                            if st.session_state.b2b_simular_llamada_clic:
                                texto_ia = f"Hola, buenas tardes. Le llamamos del Centro de Distribución {b2b_cedi} de Arca Continental. Vemos que solicitó {cantidad} cajas de {producto}, pero debido a un problema de stock, es muy probable que no las tengamos listas para despacho. ¿Aceptaría cambiarlas por {sustituto_sugerido} y le aplicamos un descuento de 8 dólares en ese artículo?"
                                
                                audio_bytes = None
                                with st.spinner("Generando audio de la llamada..."):
                                    if elevenlabs_api_key:
                                        audio_bytes = generar_voz_elevenlabs(texto_ia, elevenlabs_api_key)
                                    
                                    if audio_bytes is None:
                                        try:
                                            tts = gTTS(text=texto_ia, lang='es', tld='com.mx')
                                            temp_audio_path = os.path.join(current_dir, "temp_call_b2b.mp3")
                                            tts.save(temp_audio_path)
                                            with open(temp_audio_path, "rb") as f:
                                                audio_bytes = f.read()
                                            os.remove(temp_audio_path)
                                        except Exception as e:
                                            st.error(f"Error al generar audio de cortesía: {e}")
                                            
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                                
                                st.markdown("**Transcripción de la Conversación:**")
                                st.markdown(f"""
                                <div class='chat-bubble chat-agent'>
                                    <b>🤖 Agente IA:</b> {texto_ia}
                                </div>
                                <div class='chat-bubble chat-user'>
                                    <b>👤 Comprador ({st.session_state.b2b_cliente_nombre}):</b> Sí, está bien, acepto el cambio con descuento.
                                </div>
                                <div class='chat-bubble chat-agent'>
                                    <b>🤖 Agente IA:</b> Excelente, su pedido ha sido actualizado. Recibirá la nota corregida en unos momentos.
                                </div>
                                """, unsafe_allow_html=True)
                                
                        # Botón para confirmar cambio en la orden (simulando que el cliente lo aceptó de viva voz)
                        if st.button("✅ Aceptar Sustitución y Descuento en Orden", use_container_width=True):
                            st.session_state.b2b_llamada_confirmada = True
                            st.rerun()
            else:
                st.info("👈 Ingresa los datos de tu pedido y haz clic en 'Enviar Pedido'.")
    st.stop()

# ----------------- TABS PRINCIPALES -----------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Simulador Live Gemelo Digital + MPC", 
    "📊 Dashboard CEDI y Analíticas", 
    "📈 Optimización Bayesiana"
])

# ----------------- TAB 1: SIMULADOR LIVE -----------------
with tab1:
    st.header(f"Simulador de Pedidos: CEDI {cedi_seleccionado}")
    st.write("Simula la llegada de un pedido y observa cómo el MPC calcula la mejor respuesta operativa y cómo se activa la llamada de ElevenLabs:")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Entrada de Pedido")
        cliente = st.selectbox("Cliente B2B", ["Restaurante Centro", "Abarrotes La Esquina", "Gimnasio FitZone", "Supermercado Smart", "Taquería El Pastor", "Tienda de Conveniencia Express"])
        producto = st.selectbox("Producto Solicitado", list(inventario.keys()))
        
        avg_demand_prod = int(df_top_prods[df_top_prods['producto'] == producto]['avg_qty'].iloc[0])
        cantidad = st.number_input("Cantidad Solicitada (cajas)", min_value=1, max_value=500, value=avg_demand_prod)
        
        st.markdown("**Parámetros de Costo del MPC:**")
        penalizacion_falla = st.number_input("Penalización por Falla ($ USD)", min_value=50.0, max_value=300.0, value=120.0)
        costo_descuento = st.number_input("Costo Descuento al Cliente ($ USD)", min_value=2.0, max_value=30.0, value=8.0)
        
        procesar = st.button("🚀 Procesar Pedido", use_container_width=True)
        
        if procesar:
            if model is None:
                st.error("Error: El modelo de Machine Learning no ha sido entrenado. Corre 'entrenar_modelo.py' primero.")
            else:
                # 1. Capa ML: Predicción
                input_data = pd.DataFrame([{
                    'nombre_solicitado': producto,
                    'quantity': cantidad
                }])
                prob_desabasto = model.predict_proba(input_data)[0][1]
                
                stock_actual = inventario[producto]
                if stock_actual >= cantidad:
                    prob_real = max(0.01, prob_desabasto * 0.1)
                else:
                    deficit = cantidad - stock_actual
                    prob_real = min(0.99, prob_desabasto * 1.5 + (deficit / cantidad))
                
                # 2. Capa MPC: Comparación de Escenarios
                otros_prods = [p for p in inventario.keys() if p != producto]
                sustituto_sugerido = otros_prods[0] if otros_prods else "Coca-Cola Sin Azúcar"
                
                opciones = {
                    "A) No hacer nada (Permitir Sustitución en CEDI)": {
                        "costo_fijo": 0.0,
                        "prob_falla": prob_real,
                        "desc": "El operador toma la decisión al azar en el CEDI. Alta probabilidad de queja y devolución."
                    },
                    "B) Reubicación de stock urgente (Mover de CEDI cercano)": {
                        "costo_fijo": 35.0,
                        "prob_falla": 0.05,
                        "desc": "Envío exprés desde otro almacén. Elimina el desabasto pero tiene costo logístico."
                    },
                    "C) Notificar al cliente y autorizar sustituto con descuento": {
                        "costo_fijo": costo_descuento,
                        "prob_falla": prob_real * 0.1,
                        "desc": f"Llamada con IA de ElevenLabs al cliente ofreciendo cambiar por '{sustituto_sugerido}' con descuento."
                    }
                }
                
                nombres = []
                costos = []
                descripciones = []
                
                for nombre, opt in opciones.items():
                    costo_esperado = opt["costo_fijo"] + (opt["prob_falla"] * penalizacion_falla)
                    nombres.append(nombre)
                    costos.append(costo_esperado)
                    descripciones.append(opt["desc"])
                    
                df_mpc = pd.DataFrame({
                    "Acción Propuesta": nombres,
                    "Costo Esperado ($ USD)": costos,
                    "Descripción": descripciones
                })
                
                df_mpc = df_mpc.sort_values(by="Costo Esperado ($ USD)")
                mejor_accion = df_mpc.iloc[0]["Acción Propuesta"]
                menor_costo = df_mpc.iloc[0]["Costo Esperado ($ USD)"]
                
                st.session_state.pedido_procesado = True
                st.session_state.producto = producto
                st.session_state.cantidad = cantidad
                st.session_state.prob_real = prob_real
                st.session_state.mejor_accion = mejor_accion
                st.session_state.menor_costo = menor_costo
                st.session_state.df_mpc = df_mpc
                st.session_state.sustituto_sugerido = sustituto_sugerido
                st.session_state.simular_llamada_clic = False

    with col2:
        if st.session_state.pedido_procesado:
            st.subheader("Resultados de la Simulación")
            
            prob_real = st.session_state.prob_real
            producto = st.session_state.producto
            cantidad = st.session_state.cantidad
            mejor_accion = st.session_state.mejor_accion
            menor_costo = st.session_state.menor_costo
            df_mpc = st.session_state.df_mpc
            sustituto_sugerido = st.session_state.sustituto_sugerido
            stock_actual = inventario[producto]
            
            if prob_real > 0.6:
                st.error(f"⚠️ RIESGO CRÍTICO DE SUSTITUCIÓN: {prob_real*100:.1f}%")
            elif prob_real > 0.3:
                st.warning(f"🔸 RIESGO MODERADO DE SUSTITUCIÓN: {prob_real*100:.1f}%")
            else:
                st.success(f"✅ RIESGO BAJO DE SUSTITUCIÓN: {prob_real*100:.1f}%")
            
            st.markdown(f"""
            **Detalles de Almacén:**
            - Stock disponible en CEDI: **{stock_actual} cajas**
            - Cantidad Solicitada: **{cantidad} cajas**
            - Déficit estimado: **{max(0, cantidad - stock_actual)} cajas**
            """)
            
            st.subheader("Optimización del MPC (Model Predictive Control)")
            st.bar_chart(data=df_mpc.set_index("Acción Propuesta")["Costo Esperado ($ USD)"], color="#e41e26")
            
            st.markdown(f"""
            <div style='background-color: #ffebe6; padding: 15px; border-radius: 8px; border-left: 6px solid #e41e26; margin-bottom: 20px; color: #222222;'>
                <h4 style='margin: 0; color: #e41e26; font-weight: bold;'>💡 Recomendación de Acción Óptima:</h4>
                <p style='margin: 5px 0 0 0; font-size: 16px; font-weight: bold; color: #c62828;'>{mejor_accion}</p>
                <p style='margin: 5px 0 0 0; color: #222222;'>Costo Esperado Minimizado: <b>${menor_costo:.2f} USD</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # La simulación interactiva de la llamada se realiza únicamente en el Portal del Comprador B2B.
        else:
            st.info("👈 Selecciona los datos del pedido y haz clic en 'Procesar Pedido' para simular.")

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

# ----------------- TAB 3: OPTIMIZACION BAYESIANA -----------------
with tab3:
    st.header("Auto-Ajuste mediante Optimización Bayesiana")
    st.write("""
    El controlador MPC requiere parámetros económicos calibrados de forma precisa. 
    La **Optimización Bayesiana** (basada en Procesos Gaussianos) evalúa la simulación histórica de 100 pedidos 
    de este almacén y busca automáticamente la combinación óptima de parámetros que minimiza las pérdidas operativas.
    """)
    
    col_opt1, col_opt2 = st.columns([1, 2])
    
    with col_opt1:
        st.subheader("Control del Optimizador")
        num_iter = st.slider("Número de Iteraciones Bayesianas", 10, 30, 15)
        iniciar_opt = st.button("📈 Iniciar Búsqueda Bayesiana", use_container_width=True)
        
    with col_opt2:
        if iniciar_opt:
            with st.spinner("Ejecutando algoritmo bayesiano con Procesos Gaussianos..."):
                time.sleep(1)
                df_hist, best_x, best_y = ejecutar_optimizacion_bayesiana(current_dir, max_iter=num_iter)
                
                st.success("¡Optimización completada con éxito!")
                
                st.markdown(f"""
                ### 🏆 Configuración Óptima Encontrada:
                - **Penalización por Falla Recomendada:** `${best_x[0]:.2f} USD`
                - **Costo de Descuento Recomendado:** `${best_x[1]:.2f} USD`
                - **Costo Promedio Operativo por Pedido:** `${best_y:.2f} USD` *(Minimizado desde un promedio inicial de $15+)*.
                """)
                
                st.subheader("Curva de Aprendizaje y Minimización del Costo")
                st.line_chart(data=df_hist.set_index("Iteración")["Costo Promedio Pedido ($)"], color="#e41e26")
                
                st.subheader("Historial de Búsqueda del Proceso Gaussiano")
                st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("💡 Haz clic en 'Iniciar Búsqueda Bayesiana' para calcular la sintonización automática de parámetros.")
