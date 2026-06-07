import numpy as np
import pandas as pd
import sqlite3
import os

# GP regression desde cero con RBF Kernel
class SimpleGP:
    def __init__(self, l=30.0, noise=1.0):
        self.l = l
        self.noise = noise
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        # Centramos y para estabilidad
        self.y_mean = np.mean(y)
        self.y_train = np.array(y) - self.y_mean
        
    def predict(self, X_test):
        X_test = np.array(X_test)
        if self.X_train is None or len(self.X_train) == 0:
            return np.zeros(len(X_test)), np.ones(len(X_test))
            
        K = self._kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        K_s = self._kernel(X_test, self.X_train)
        K_ss = self._kernel(X_test, X_test) + self.noise * np.eye(len(X_test))
        
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K) # pseudo-inversa por seguridad
            
        mu = K_s.dot(K_inv).dot(self.y_train) + self.y_mean
        sigma2 = np.diag(K_ss - K_s.dot(K_inv).dot(K_s.T))
        return mu, np.sqrt(np.maximum(1e-8, sigma2))
        
    def _kernel(self, X1, X2):
        # Distancia euclidiana al cuadrado para RBF kernel
        d1 = np.sum(X1**2, axis=1).reshape(-1, 1)
        d2 = np.sum(X2**2, axis=1)
        dist_matrix = d1 + d2 - 2 * np.dot(X1, X2.T)
        return np.exp(-dist_matrix / (2 * self.l**2))

# Función objetivo: Simula el costo total de procesar 100 pedidos
# dadas las configuraciones de parámetros del MPC: (penalizacion_falla, costo_descuento)
def evaluar_costo_mpc(penalizacion_falla, costo_descuento, orders_sample):
    total_cost = 0.0
    
    # Costos fijos del MPC
    costos_opciones = {
        "A": {"costo_fijo": 0.0, "prob_falla_factor": 1.0},
        "B": {"costo_fijo": 35.0, "prob_falla_factor": 0.0}, # Reubicación siempre exitosa en esta simulación
        "C": {"costo_fijo": costo_descuento, "prob_falla_factor": 0.1}
    }
    
    # Para simular, usamos un generador de números aleatorios con semilla fija 
    # para que la evaluación sea determinista en cada iteración del optimizador.
    rng = np.random.default_rng(42)
    
    for _, row in orders_sample.iterrows():
        # Riesgo base simulado de desabasto del producto
        prob_base = rng.uniform(0.1, 0.9)
        
        # El MPC toma decisiones evaluando costos esperados
        mejor_accion = None
        menor_costo_esperado = float('inf')
        
        for accion, opt in costos_opciones.items():
            prob_falla = prob_base * opt["prob_falla_factor"]
            costo_esperado = opt["costo_fijo"] + (prob_falla * penalizacion_falla)
            
            if costo_esperado < menor_costo_esperado:
                menor_costo_esperado = costo_esperado
                mejor_accion = accion
                
        # Simulación de la ejecución de la acción decidida:
        # Evaluamos el costo real cobrado (costo fijo de la opción + penalización si falla realmente)
        opt_elegida = costos_opciones[mejor_accion]
        falla_real = rng.random() < (prob_base * opt_elegida["prob_falla_factor"])
        
        costo_real = opt_elegida["costo_fijo"]
        if falla_real:
            costo_real += 120.0 # Penalización real de insatisfacción del cliente
            
        total_cost += costo_real
        
    return total_cost / len(orders_sample) # Costo promedio por pedido

def ejecutar_optimizacion_bayesiana(base_dir, max_iter=15):
    db_path = os.path.join(base_dir, "order_rescue.db")
    if not os.path.exists(db_path):
        # Si no hay DB creamos una muestra artificial
        print("Base de datos no encontrada. Creando muestra sintética para simular...")
        orders_sample = pd.DataFrame([{"id": i} for i in range(100)])
    else:
        conn = sqlite3.connect(db_path)
        # Cargamos una muestra de 100 pedidos
        orders_sample = pd.read_sql_query("SELECT id_pedido FROM orders LIMIT 100;", conn)
        conn.close()
        
    # Inicialización del optimizador
    # Rangos a optimizar:
    # 1. Penalización Falla (x1): [50, 200]
    # 2. Costo Descuento (x2): [2, 20]
    
    # 3 Puntos iniciales aleatorios
    X_sample = [
        [60.0, 5.0],
        [180.0, 18.0],
        [120.0, 10.0]
    ]
    y_sample = [evaluar_costo_mpc(x[0], x[1], orders_sample) for x in X_sample]
    
    historia_opt = []
    for i, (x, y) in enumerate(zip(X_sample, y_sample)):
        historia_opt.append({
            "Iteración": i + 1,
            "Penalización Falla ($)": x[0],
            "Costo Descuento ($)": x[1],
            "Costo Promedio Pedido ($)": y,
            "Tipo": "Inicial"
        })
        
    gp = SimpleGP()
    
    # Crear grilla densa para la función de adquisición (1600 puntos)
    x1_grid = np.linspace(50.0, 200.0, 40)
    x2_grid = np.linspace(2.0, 20.0, 40)
    X_grid = np.array([[x1, x2] for x1 in x1_grid for x2 in x2_grid])
    
    # Loop de Optimización Bayesiana
    for it in range(len(X_sample), max_iter):
        gp.fit(X_sample, y_sample)
        
        # Predicción sobre la grilla
        mu, sigma = gp.predict(X_grid)
        
        # Criterio de adquisición LCB (Lower Confidence Bound) para MINIMIZAR costo
        # LCB = mu - beta * sigma
        beta = 2.0
        lcb = mu - beta * sigma
        
        # Siguiente punto es el que minimiza el LCB
        next_idx = np.argmin(lcb)
        x_next = X_grid[next_idx].tolist()
        
        # Evitar evaluar duplicados exactos
        if any(np.allclose(x_next, x, atol=1.0) for x in X_sample):
            # Añadir pequeña perturbación
            x_next[0] += np.random.uniform(-5, 5)
            x_next[1] += np.random.uniform(-1, 1)
            # Clip en rangos
            x_next[0] = np.clip(x_next[0], 50.0, 200.0)
            x_next[1] = np.clip(x_next[1], 2.0, 20.0)
            
        y_next = evaluar_costo_mpc(x_next[0], x_next[1], orders_sample)
        
        X_sample.append(x_next)
        y_sample.append(y_next)
        
        historia_opt.append({
            "Iteración": it + 1,
            "Penalización Falla ($)": x_next[0],
            "Costo Descuento ($)": x_next[1],
            "Costo Promedio Pedido ($)": y_next,
            "Tipo": "Bayesiano"
        })
        
    df_historia = pd.DataFrame(historia_opt)
    
    # Encontrar mejor resultado
    best_idx = np.argmin(y_sample)
    best_x = X_sample[best_idx]
    best_y = y_sample[best_idx]
    
    print("\n--- Resultados de la Optimización Bayesiana ---")
    print(f"Mejor iteración: {best_idx + 1}")
    print(f"Parámetros Óptimos:")
    print(f"  - Penalización Falla: ${best_x[0]:.2f} USD")
    print(f"  - Costo Descuento: ${best_x[1]:.2f} USD")
    print(f"Costo Promedio Mínimo por Pedido: ${best_y:.2f} USD")
    
    # Guardar resultados
    df_historia.to_csv(os.path.join(base_dir, "historial_optimizacion.csv"), index=False)
    print("Guardado historial_optimizacion.csv")
    
    return df_historia, best_x, best_y

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ejecutar_optimizacion_bayesiana(current_dir)
