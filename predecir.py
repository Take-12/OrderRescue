import os
import pickle
import pandas as pd

def predecir_sustitucion():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "modelo_order_rescue.pkl")
    
    if not os.path.exists(model_path):
        print("Error: El modelo no ha sido entrenado aún. Ejecuta primero 'entrenar_modelo.py'.")
        return
        
    # Cargamos el modelo
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    print("==================================================")
    print("      Simulador de Predicción: Order Rescue       ")
    print("==================================================")
    print("Ingresa los datos del pedido para calcular la probabilidad de sustitución.\n")
    
    while True:
        producto = input("Nombre del producto solicitado (o escribe 'salir' para terminar): ").strip()
        if producto.lower() == 'salir':
            break
            
        if not producto:
            continue
            
        try:
            cantidad_str = input("Cantidad solicitada: ").strip()
            cantidad = int(cantidad_str)
        except ValueError:
            print("Por favor ingresa un número entero válido para la cantidad.\n")
            continue
            
        # Creamos DataFrame de entrada
        input_data = pd.DataFrame([{
            'nombre_solicitado': producto,
            'quantity': cantidad
        }])
        
        # Predicción
        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]
        
        print("\n---------------- Resultados ----------------")
        print(f"Producto: {producto}")
        print(f"Cantidad: {cantidad}")
        print(f"Probabilidad de Sustitución: {prob*100:.2f}%")
        if prob > 0.5:
            print(">>> ALERTA: Alto riesgo de sustitución.")
        else:
            print(">>> Estado: Bajo riesgo de sustitución.")
        print("--------------------------------------------\n")

if __name__ == "__main__":
    predecir_sustitucion()
