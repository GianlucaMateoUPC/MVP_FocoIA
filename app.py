# ==========================================
# ✅ Foco IA (compatible con Streamlit Cloud)
# ==========================================

import streamlit as st
import threading
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import sqlite3
import bcrypt
import json

# ==========================================
# ✅ Token y cliente HF
# ==========================================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HF_TOKEN
)

# ==========================================
# ✅ IA de enfoque (imagen simulada)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mobilenet_foco_distraccion.pth")
EJEMPLO_IMG = os.path.join(BASE_DIR, "ejemplo_foco.png")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ['distraccion', 'foco']

# ==========================================
# ✅ Estado global robusto
# ==========================================
if "monitoreo_event" not in st.session_state:
    st.session_state.monitoreo_event = threading.Event()
    st.session_state.monitoreo_activo = False

if "estado_emocional" not in st.session_state:
    st.session_state.estado_emocional = 3

if "ultimo_checkin" not in st.session_state:
    st.session_state.ultimo_checkin = None

if "ultima_prediccion" not in st.session_state:
    st.session_state.ultima_prediccion = "N/A"

if "mostrar_chat" not in st.session_state:
    st.session_state.mostrar_chat = False

if "puntos_totales" not in st.session_state:
    st.session_state.puntos_totales = 0

if "racha_dias" not in st.session_state:
    st.session_state.racha_dias = 0

if "ultima_fecha_punto" not in st.session_state:
    st.session_state.ultima_fecha_punto = ""

dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
horas = [f"{h} AM" if h < 12 else f"{h-12} PM" if h > 12 else "12 PM" for h in range(7, 24)]

if "subtareas_dict" not in st.session_state:
    st.session_state.subtareas_dict = {}

# ==========================================
# ✅ Función monitoreo IA enfoque (simulada)
# ==========================================
def monitoreo_ia(evento):
    tiempo_foco = 0
    while not evento.is_set():
        screenshot = Image.open(EJEMPLO_IMG)
        image = transform(screenshot).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            clase = classes[predicted.item()]
        st.session_state.ultima_prediccion = clase

        if clase == "distraccion":
            st.warning("🚨 ¡Detecté distracción! Haz click en el chat para apoyo 👇")

        tiempo_foco += 10
        if tiempo_foco >= 60:
            st.info("⏸️ Pausa Activa: Relaja tus ojos y estírate.")
            st.balloons()
            evento.set()
            time.sleep(60)
            st.session_state.monitoreo_event = threading.Event()
            hilo_nuevo = threading.Thread(target=monitoreo_ia, args=(st.session_state.monitoreo_event,), daemon=True)
            hilo_nuevo.start()
            st.session_state.monitoreo_activo = True
            return
        time.sleep(10)
    st.info("✅ Monitoreo IA detenido.")

# ==========================================
# ✅ Configuración visual
# ==========================================
st.set_page_config(page_title="Foco IA + Zephyr 7B", layout="wide")

# ======================
# 🔐 LOGIN DE USUARIOS
# ======================
def verificar_credenciales(correo, contrasena):
    conn = sqlite3.connect("usuarios.db")
    cursor = conn.cursor()
    cursor.execute("SELECT contrasena_hash FROM usuarios WHERE correo = ?", (correo,))
    resultado = cursor.fetchone()
    conn.close()
    if resultado:
        hash_guardado = resultado[0]
        return bcrypt.checkpw(contrasena.encode('utf-8'), hash_guardado)
    return False

def registrar_usuario(correo, contrasena):
    conn = sqlite3.connect("usuarios.db")
    cursor = conn.cursor()
    contrasena_hash = bcrypt.hashpw(contrasena.encode('utf-8'), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO usuarios (correo, contrasena_hash) VALUES (?, ?)", (correo, contrasena_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

if "logueado" not in st.session_state:
    st.session_state.logueado = False
if "modo_registro" not in st.session_state:
    st.session_state.modo_registro = False

if not st.session_state.logueado:
    if st.session_state.modo_registro:
        st.title("📝 Registro de nuevo usuario")
        nuevo_correo = st.text_input("Correo nuevo")
        nueva_contrasena = st.text_input("Contraseña nueva", type="password")
        if st.button("Registrar"):
            if registrar_usuario(nuevo_correo, nueva_contrasena):
                st.success("✅ Registro exitoso. Ahora inicia sesión.")
                st.session_state.modo_registro = False
                st.rerun()
            else:
                st.error("❌ El correo ya está registrado.")
        if st.button("⬅ Volver al login"):
            st.session_state.modo_registro = False
            st.rerun()
    else:
        st.title("🔐 Iniciar sesión")
        correo = st.text_input("Correo")
        contrasena = st.text_input("Contraseña", type="password")
        if st.button("Ingresar"):
            if verificar_credenciales(correo, contrasena):
                st.session_state.logueado = True
                st.session_state.usuario_correo = correo

                conn = sqlite3.connect("usuarios.db")
                cursor = conn.cursor()
                cursor.execute("SELECT horario FROM usuarios WHERE correo = ?", (correo,))
                resultado = cursor.fetchone()
                conn.close()

                if resultado and resultado[0]:
                    try:
                        horario_cargado = pd.read_json(resultado[0])
                        st.session_state.horario_grid = horario_cargado
                    except Exception:
                        st.session_state.horario_grid = pd.DataFrame(
                            np.full((len(horas), len(dias_semana)), "", dtype=object),
                            index=horas,
                            columns=dias_semana
                        )
                else:
                    st.session_state.horario_grid = pd.DataFrame(
                        np.full((len(horas), len(dias_semana)), "", dtype=object),
                        index=horas,
                        columns=dias_semana
                    )

                st.success("✅ Bienvenido a Foco IA")
                st.rerun()
            else:
                st.error("❌ Credenciales incorrectas")
        if st.button("🆕 ¿No tienes cuenta? Regístrate aquí"):
            st.session_state.modo_registro = True
            st.rerun()
    st.stop()

# ==========================================
# ✅ Horario editable y persistente
# ==========================================
def guardar_horario_en_db(correo, horario_df):
    conn = sqlite3.connect("usuarios.db")
    cursor = conn.cursor()
    horario_str = horario_df.to_json()
    cursor.execute("UPDATE usuarios SET horario = ? WHERE correo = ?", (horario_str, correo))
    conn.commit()
    conn.close()

# ==========================================
# ✅ Selector de tema visual con fondo simulado
# ==========================================
tema = st.sidebar.selectbox(
    "🎨 Selecciona tu estilo visual:",
    ["Relajante (verde/azul)", "Energizante (naranja/rojo)", "Nocturno (oscuro)"]
)

# Fondo simulado por HTML
def aplicar_fondo_css(css_background, css_text_color):
    st.markdown(f"""
        <style>
        body {{
            background: {css_background} !important;
            color: {css_text_color} !important;
        }}
        .stApp {{
            background: {css_background} !important;
        }}
        .block-container {{
            background: transparent !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# Aplicar el fondo según el tema
if tema == "Relajante (verde/azul)":
    aplicar_fondo_css("linear-gradient(to right, #a8edea, #fed6e3)", "#004d40")
elif tema == "Energizante (naranja/rojo)":
    aplicar_fondo_css("linear-gradient(to right, #fbd786, #f7797d)", "#4e0303")
elif tema == "Nocturno (oscuro)":
    aplicar_fondo_css("#0e1117", "#fafafa")

st.title("🎯 Foco IA + Chatbot Zephyr 7B")

col1, col2 = st.columns(2)

with col1:
    if st.button("Iniciar sesión de foco"):
        st.session_state.monitoreo_event.clear()
        hilo = threading.Thread(target=monitoreo_ia,
                                args=(st.session_state.monitoreo_event,),
                                daemon=True)
        hilo.start()
        st.session_state.monitoreo_activo = True
        st.success("✅ Monitoreo IA activo")

with col2:
    if st.button("Detener monitoreo"):
        st.session_state.monitoreo_event.set()
        st.session_state.monitoreo_activo = False
        st.success("✅ Monitoreo IA detenido")

st.info(f"🧠 Última predicción IA: **{st.session_state.ultima_prediccion.upper()}**")

# ==========================================
# ✅ Check-in emocional
# ==========================================
st.header("🧘 Check-in emocional")
estado = st.slider("¿Cómo te sientes ahora? (1 = Mal, 5 = Excelente)", 1, 5, st.session_state.estado_emocional)
ahora = datetime.now()
habilitado = st.session_state.ultimo_checkin is None or (ahora - st.session_state.ultimo_checkin > timedelta(minutes=2))

if st.button("Registrar estado emocional", disabled=not habilitado):
    st.session_state.estado_emocional = estado
    st.session_state.ultimo_checkin = ahora
    st.success(f"👍 Estado registrado: {estado}/5")

if not habilitado:
    restante = timedelta(minutes=2) - (ahora - st.session_state.ultimo_checkin)
    st.info(f"⏳ Próximo check-in disponible en: {restante.seconds} segundos.")

# ==========================================
# ✅ Horario editable
# ==========================================
st.header("📅 Horario de clases")
if st.button("Importar horario (.csv)"):
    st.info("✨ (Demo) Importación simulada.")
st.info("✏️ Doble click en cada celda para editar tu horario")
st.session_state.horario_grid = st.data_editor(
    st.session_state.horario_grid,
    use_container_width=True,
    num_rows="dynamic"
)

if st.session_state.get("usuario_correo"):
    guardar_horario_en_db(st.session_state.usuario_correo, st.session_state.horario_grid)

# ==========================================
# ✅ Tareas con subtareas dinámicas
# ==========================================
st.header("📋 Tareas con subtareas y progreso")
tareas_principales = []
for dia in dias_semana:
    for hora in horas:
        contenido = st.session_state.horario_grid.loc[hora, dia]
        if isinstance(contenido, str) and contenido.strip():
            tareas_principales.append(f"{contenido} ({dia} {hora})")

for tarea in tareas_principales:
    if tarea not in st.session_state.subtareas_dict:
        st.session_state.subtareas_dict[tarea] = {"subtareas": ["Subtarea 1"]}
for tarea in tareas_principales:
    st.subheader(f"📌 {tarea}")
    subtareas = st.session_state.subtareas_dict[tarea]["subtareas"]
    completados = 0
    to_delete = []
    for idx, sub in enumerate(subtareas):
        cols = st.columns([0.7, 0.1, 0.1])
        new_name = cols[0].text_input(f"Subtarea:", value=sub, key=f"{tarea}-name-{idx}")
        st.session_state.subtareas_dict[tarea]["subtareas"][idx] = new_name
        done = cols[1].checkbox("✅", key=f"{tarea}-done-{idx}")
        if done:
            completados += 1
        delete = cols[2].button("❌", key=f"{tarea}-del-{idx}")
        if delete:
            to_delete.append(idx)
    for idx in sorted(to_delete, reverse=True):
        subtareas.pop(idx)
    if st.button(f"➕ Agregar subtarea", key=f"{tarea}-add"):
        subtareas.append(f"Subtarea {len(subtareas)+1}")
    progreso = completados / len(subtareas) if subtareas else 0
    st.progress(progreso)

# ==========================================
# ✅ Sistema de Puntos y Racha (Gamificación robusta)
# ==========================================
st.header("🏅 Progreso y Motivación Diaria")

hoy = datetime.now().strftime("%Y-%m-%d")

acciones = {
    "Realizaste un check-in emocional ✅": hoy == st.session_state.ultimo_checkin.strftime("%Y-%m-%d") if st.session_state.ultimo_checkin else False,
    "Monitoreo activo por +30 segundos ⏱️": st.session_state.monitoreo_activo
}

puntos_ganados = 0
for accion, cumplida in acciones.items():
    if cumplida:
        st.write(f"✔ {accion} (+10 pts)")
        puntos_ganados += 10

if puntos_ganados > 0:
    if st.session_state.ultima_fecha_punto != hoy:
        st.session_state.racha_dias += 1
        st.session_state.ultima_fecha_punto = hoy
    st.session_state.puntos_totales += puntos_ganados

st.success(f"🎯 Puntos acumulados: **{st.session_state.puntos_totales}**")
st.info(f"🔥 Racha actual: **{st.session_state.racha_dias} días seguidos**")
st.progress(min(st.session_state.racha_dias / 7, 1.0), text="¡Sigue así! A los 7 días desbloqueas un logro 🎉")

# ==========================================
# ✅ Chatbot REAL con Zephyr 7B + prompt estructurado
# ==========================================
st.header("💬 Chatbot Zephyr 7B")

if st.button("Abrir Chatbot"):
    st.session_state.mostrar_chat = not st.session_state.mostrar_chat

if st.session_state.mostrar_chat:
    st.write("🤖 Hola, soy Zephyr 7B. Escribe tu mensaje (responderé en español):")
    pregunta = st.text_input("Tu mensaje:", key="input_user")
    if pregunta:
        with st.spinner("Pensando..."):
            messages = [
                {"role": "system", "content": "Eres un chatbot empático, motivacional y SIEMPRE respondes en español."},
                {"role": "user", "content": pregunta}
            ]
            respuesta = client.chat_completion(messages=messages, max_tokens=200)
            st.write(f"Chatbot: {respuesta.choices[0].message['content']}")