import sqlite3
import bcrypt

# Conexión a la base de datos
conn = sqlite3.connect("usuarios.db")
cursor = conn.cursor()

# Crear tabla usuarios
cursor.execute('''
CREATE TABLE IF NOT EXISTS usuarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    correo TEXT UNIQUE NOT NULL,
    contrasena_hash TEXT NOT NULL
)
''')

# Función para insertar usuario
def insertar_usuario(correo, contrasena):
    contrasena_hash = bcrypt.hashpw(contrasena.encode('utf-8'), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO usuarios (correo, contrasena_hash) VALUES (?, ?)", (correo, contrasena_hash))
        conn.commit()
        print(f"✅ Usuario '{correo}' creado correctamente.")
    except sqlite3.IntegrityError:
        print(f"⚠️ El usuario '{correo}' ya existe.")

# Usuarios de ejemplo
insertar_usuario("usuario1@example.com", "clave123")
insertar_usuario("usuario2@example.com", "miclave456")

conn.close()