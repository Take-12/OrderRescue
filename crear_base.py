import pandas as pd
import sqlite3
import os

def crear_base_de_datos(base_dir):
    db_path = os.path.join(base_dir, "order_rescue.db")
    
    # Eliminamos la base de datos previa si existe para hacerla limpia
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Eliminada base de datos previa.")
        
    conn = sqlite3.connect(db_path)
    
    archivos = {
        "orders": os.path.join(base_dir, "Orders.csv"),
        "order_details": os.path.join(base_dir, "OrderDetails.csv"),
        "sustituciones": os.path.join(base_dir, "Resultados.csv")
    }
    
    # Columnas que queremos forzar como texto
    cols_como_texto = {
        "id_pedido", "customer_id", "id_linea", "sku_solicitado", 
        "sku_solicitado_hash", "sku_solicitado_cambio", 
        "sku_solicitado_cambio_hash", "cedis"
    }
    
    print("Iniciando creación de la base de datos con Strings sin truncar...")
    
    for tabla, archivo in archivos.items():
        if not os.path.exists(archivo):
            print(f"Error: No se encontró el archivo {archivo}.")
            continue
            
        print(f"Procesando {archivo}...")
        
        # Leemos encabezado para ver las columnas
        header = pd.read_csv(archivo, nrows=0)
        cols_limpias = [c.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_") for c in header.columns]
        
        dtype_spec = {}
        for col_original, col_limpia in zip(header.columns, cols_limpias):
            if col_limpia in cols_como_texto:
                dtype_spec[col_original] = str
                
        first = True
        total = 0
        
        try:
            for chunk in pd.read_csv(archivo, chunksize=100000, dtype=dtype_spec, low_memory=False):
                # Limpiamos nombres de columnas
                chunk.columns = (
                    chunk.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_", regex=False)
                    .str.replace("-", "_", regex=False)
                    .str.replace(".", "_", regex=False)
                )
                
                # Normalizamos las columnas de texto (quitar espacios y nulos)
                for col in chunk.columns:
                    if col in cols_como_texto:
                        chunk[col] = chunk[col].fillna("").astype(str).str.strip()
                        # Si quedó la palabra "nan", la limpiamos a vacío
                        chunk[col] = chunk[col].replace("nan", "")
                        
                chunk.to_sql(
                    tabla,
                    conn,
                    if_exists="replace" if first else "append",
                    index=False
                )
                total += len(chunk)
                first = False
            print(f"Tabla '{tabla}' creada con {total} registros.")
        except Exception as e:
            print(f"Error procesando {tabla}: {e}")
            
    conn.commit()
    conn.close()
    print(f"Base de datos finalizada en: {db_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    crear_base_de_datos(current_dir)
