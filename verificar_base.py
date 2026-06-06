import sqlite3
import pandas as pd
import os

def verificar_base(base_dir):
    db_path = os.path.join(base_dir, "order_rescue.db")
    if not os.path.exists(db_path):
        print("Error: La base de datos no existe.")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Obtener tablas
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tablas = cursor.fetchall()
    print("Tablas encontradas:", [t[0] for t in tablas])
    
    for (tabla,) in tablas:
        print(f"\n--- Estructura y muestra de la tabla: {tabla} ---")
        # Mostrar columnas
        cursor.execute(f"PRAGMA table_info({tabla});")
        columnas = cursor.fetchall()
        print("Columnas:")
        for col in columnas:
            print(f"  - {col[1]} ({col[2]})")
            
        # Mostrar primeros 3 registros
        df = pd.read_sql_query(f"SELECT * FROM {tabla} LIMIT 3;", conn)
        print("Muestra de datos (primeras 3 filas):")
        print(df.to_string())
        
        # Conteo total
        cursor.execute(f"SELECT count(*) FROM {tabla};")
        total = cursor.fetchone()[0]
        print(f"Total registros: {total}")
        
    conn.close()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    verificar_base(current_dir)
