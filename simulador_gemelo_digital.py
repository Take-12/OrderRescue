import os
import pickle
import pandas as pd
import numpy as np

def run_digital_twin_simulation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "modelo_order_rescue.pkl")
    
    if not os.path.exists(model_path):
        print("Error: El modelo no ha sido entrenado aún. Ejecuta primero 'entrenar_modelo.py'.")
        return
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    print("======================================================================")
    print("    Simulador de Gemelo Digital + MPC (Model Predictive Control)       ")
    print("======================================================================")
    
    # 1. Definimos el estado del inventario inicial del CEDI (Monterrey)
    inventario = {
        "Ciel Agua Purificada": 15,
        "Coca - Cola": 120,
        "Powerade Moras": 8,
        "Topo Chico Agua Mineral": 12,
        "Yogurt Toni Durazno 110 Gr.": 5
    }
    
    print("\n[Gemelo Digital] Estado del Inventario Real en CEDI:")
    for prod, stock in inventario.items():
        print(f"  - {prod}: {stock} cajas en stock")
        
    # 2. Definimos una lista de pedidos entrantes a simular
    pedidos_simulados = [
        {"cliente": "Restaurante Centro", "producto": "Ciel Agua Purificada", "cantidad": 20},
        {"cliente": "Abarrotes La Esquina", "producto": "Coca - Cola", "cantidad": 30},
        {"cliente": "Gimnasio FitZone", "producto": "Powerade Moras", "cantidad": 10},
        {"cliente": "Supermercado Smart", "producto": "Topo Chico Agua Mineral", "cantidad": 25},
        {"cliente": "Taquería El Pastor", "producto": "Coca - Cola", "cantidad": 150}
    ]
    
    print(f"\n[Gemelo Digital] Recibidos {len(pedidos_simulados)} pedidos para procesar. Simulando decisión del MPC...\n")
    
    for idx, ped in enumerate(pedidos_simulados):
        print(f"--- Procesando Pedido #{idx+1} ({ped['cliente']}) ---")
        print(f"  Solicitado: {ped['cantidad']} cajas de '{ped['producto']}'")
        
        # 1. Capa ML: Predecimos riesgo de sustitución
        input_data = pd.DataFrame([{
            'nombre_solicitado': ped['producto'],
            'quantity': ped['cantidad']
        }])
        prob_desabasto = model.predict_proba(input_data)[0][1]
        
        # Ajustamos la probabilidad en base a nuestro inventario actual
        stock_actual = inventario.get(ped['producto'], 0)
        if stock_actual >= ped['cantidad']:
            # Hay stock suficiente, la probabilidad real de desabasto es baja
            prob_real = max(0.01, prob_desabasto * 0.1)
        else:
            # Falta stock
            deficit = ped['cantidad'] - stock_actual
            prob_real = min(0.99, prob_desabasto * 1.5 + (deficit / ped['cantidad']))
            
        print(f"  [ML] Probabilidad Calculada de Desabasto: {prob_real*100:.2f}%")
        
        # 2. Capa MPC: Evaluamos decisiones basándonos en costo esperado
        # Definimos costos y riesgos de cada opción
        opciones = {
            "A) No hacer nada (Permitir Sustitución en CEDI)": {
                "costo_fijo": 0,
                "penalizacion_falla": 120, # Alto costo de insatisfacción y devolución
                "prob_falla": prob_real
            },
            "B) Reubicación de stock urgente (Mover de CEDI cercano)": {
                "costo_fijo": 35, # Costo logístico del flete express
                "penalizacion_falla": 120,
                "prob_falla": 0.05 # Riesgo de que no llegue a tiempo
            },
            "C) Notificar al cliente y autorizar sustituto con descuento": {
                "costo_fijo": 8, # Costo del descuento promocional ofrecido
                "penalizacion_falla": 120,
                "prob_falla": prob_real * 0.1 # Muy bajo riesgo de queja ya que el cliente lo aprobó
            }
        }
        
        mejor_opcion = None
        menor_costo_esperado = float('inf')
        
        print("\n  [MPC] Evaluando escenarios de costos esperados:")
        for nombre, opt in opciones.items():
            # Costo esperado = Costo Fijo + (Probabilidad de Falla * Penalización por Falla)
            costo_esperado = opt["costo_fijo"] + (opt["prob_falla"] * opt["penalizacion_falla"])
            print(f"    * {nombre}: Costo Esperado = ${costo_esperado:.2f} USD")
            
            if costo_esperado < menor_costo_esperado:
                menor_costo_esperado = costo_esperado
                mejor_opcion = nombre
                
        print(f"\n  >>> [MPC RECOMENDACIÓN] Seleccionada: {mejor_opcion} (Costo Esperado: ${menor_costo_esperado:.2f} USD)")
        print("----------------------------------------------------------------------\n")

if __name__ == "__main__":
    run_digital_twin_simulation()
