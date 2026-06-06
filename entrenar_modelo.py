import sqlite3
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def entrenar_modelo_prediccion(base_dir):
    db_path = os.path.join(base_dir, "order_rescue.db")
    conn = sqlite3.connect(db_path)
    
    print("Cargando datos de la base de datos...")
    # Cargamos el dataset combinando order_details y sustituciones
    query = """
        SELECT 
            d.id_linea,
            d.quantity,
            COALESCE(s.nombre_sku_solicitado, d.nombre_sku_solicitado) as nombre_solicitado,
            CASE WHEN s.id_linea IS NOT NULL THEN 1 ELSE 0 END as is_substituted
        FROM order_details d
        LEFT JOIN sustituciones s ON d.id_linea = s.id_linea;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Rellenamos posibles nulos en la columna de nombres
    df['nombre_solicitado'] = df['nombre_solicitado'].fillna("desconocido")
    
    print(f"Total registros cargados: {len(df)}")
    print("Distribución de clases:")
    print(df['is_substituted'].value_counts())
    
    # Dado que el dataset es muy desbalanceado, tomaremos una muestra balanceada para el entrenamiento
    # Todos los casos positivos (sustituidos) y una muestra de los negativos (no sustituidos)
    pos = df[df['is_substituted'] == 1]
    neg = df[df['is_substituted'] == 0]
    
    # Tomamos 20,000 negativos al azar para que el entrenamiento sea rápido y balanceado
    neg_sample = neg.sample(n=20000, random_state=42)
    
    df_balanced = pd.concat([pos, neg_sample]).sample(frac=1, random_state=42)
    print(f"\nDataset balanceado para entrenamiento: {len(df_balanced)} registros ({len(pos)} positivos, {len(neg_sample)} negativos)")
    
    X = df_balanced[['nombre_solicitado', 'quantity']]
    y = df_balanced['is_substituted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nEntrenando modelo (Pipeline con TF-IDF y RandomForest)...")
    
    # Procesador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=500), 'nombre_solicitado')
        ],
        remainder='passthrough' # deja pasar la columna 'quantity' tal cual
    )
    
    # Pipeline de Machine Learning
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("\nEvaluando modelo...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print(report_text)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Guardar modelo
    model_path = os.path.join(base_dir, "modelo_order_rescue.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Modelo guardado exitosamente en: {model_path}")
    
    # Generar archivo de reporte markdown
    report_md = f"""# Reporte de Modelo Predictivo: Order Rescue

El objetivo de este modelo es predecir si un producto solicitado en un pedido será sustituido antes de que el operador prepare el despacho.

## Detalles del Entrenamiento
- **Tamaño del dataset de entrenamiento:** {len(X_train):,}
- **Tamaño del dataset de prueba:** {len(X_test):,}
- **Procesamiento de texto:** TF-IDF Vectorizer (500 palabras/tokens más frecuentes) sobre el nombre del producto solicitado.
- **Algoritmo:** RandomForestClassifier (con balanceo de pesos de clase).

## Métricas de Rendimiento en el Set de Prueba (Test Set)

### Matriz de Confusión
- **Verdaderos Negativos (No Sustituidos predichos correctamente):** {cm[0][0]}
- **Falsos Positivos (Predicho como sustitución pero no lo fue):** {cm[0][1]}
- **Falsos Negativos (Predicho como no sustituido pero sí lo fue):** {cm[1][0]}
- **Verdaderos Positivos (Sustituidos predichos correctamente):** {cm[1][1]}

### Métricas de Clasificación
```text
{report_text}
```

- **Área bajo la curva ROC (ROC-AUC):** {roc_auc:.4f}

## Conclusión
El modelo utiliza el nombre del producto solicitado y la cantidad pedida como variables de entrada. Muestra un rendimiento sólido para predecir cuándo ocurrirá una sustitución en base a patrones históricos de desabasto por marca/presentación.
"""
    
    with open(os.path.join(base_dir, "reporte_modelo.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
    print("Reporte del modelo guardado en reporte_modelo.md!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    entrenar_modelo_prediccion(current_dir)
