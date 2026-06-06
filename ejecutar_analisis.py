import sqlite3
import pandas as pd
import os

def run_analysis(base_dir):
    db_path = os.path.join(base_dir, "order_rescue.db")
    conn = sqlite3.connect(db_path)
    
    report_content = "# Reporte de Análisis Exploratorio: Order Rescue\n\n"
    
    # 1. Total Substitutions and overall rate
    print("Calculando tasa de sustitución...")
    total_orders_query = "SELECT count(distinct id_pedido) FROM orders;"
    total_orders = conn.execute(total_orders_query).fetchone()[0]
    
    sub_orders_query = "SELECT count(distinct id_pedido) FROM sustituciones;"
    sub_orders = conn.execute(sub_orders_query).fetchone()[0]
    
    total_lines_query = "SELECT count(*) FROM order_details;"
    total_lines = conn.execute(total_lines_query).fetchone()[0]
    
    sub_lines_query = "SELECT count(*) FROM sustituciones;"
    sub_lines = conn.execute(sub_lines_query).fetchone()[0]
    
    report_content += "## Resumen General\n"
    report_content += f"- **Total de Pedidos:** {total_orders:,}\n"
    report_content += f"- **Pedidos con Sustituciones:** {sub_orders:,} ({sub_orders/total_orders*100:.2f}%)\n"
    report_content += f"- **Total de Líneas de Pedido:** {total_lines:,}\n"
    report_content += f"- **Líneas Sustituidas:** {sub_lines:,} ({sub_lines/total_lines*100:.4f}%)\n\n"
    
    # 2. Top 15 products requested that get substituted
    print("Calculando productos más sustituidos...")
    top_substituted = pd.read_sql_query("""
        SELECT 
            sku_solicitado,
            nombre_sku_solicitado as producto_solicitado,
            count(*) as total_sustituciones
        FROM sustituciones
        GROUP BY sku_solicitado, nombre_sku_solicitado
        ORDER BY total_sustituciones DESC
        LIMIT 15;
    """, conn)
    
    report_content += "## 1. Productos Solicitados Más Sustituidos (Top 15)\n"
    report_content += top_substituted.to_markdown(index=False) + "\n\n"
    
    # 3. Top 15 most common substitutions (Pairing)
    print("Calculando parejas de sustitución más comunes...")
    top_pairs = pd.read_sql_query("""
        SELECT 
            nombre_sku_solicitado as solicitado,
            nombre_sku_solicitado_cambio as sustituto,
            count(*) as total
        FROM sustituciones
        GROUP BY nombre_sku_solicitado, nombre_sku_solicitado_cambio
        ORDER BY total DESC
        LIMIT 15;
    """, conn)
    
    report_content += "## 2. Parejas de Sustitución Más Comunes (Top 15)\n"
    report_content += top_pairs.to_markdown(index=False) + "\n\n"
    
    # 4. Top CEDIS with most substitutions
    print("Calculando CEDIS con más sustituciones...")
    # We need to join orders with sustituciones
    top_cedis = pd.read_sql_query("""
        SELECT 
            o.cedis,
            o.pais,
            count(distinct s.id_pedido) as pedidos_con_sustitucion,
            count(s.id_linea) as total_sustituciones
        FROM orders o
        JOIN sustituciones s ON o.id_pedido = s.id_pedido
        GROUP BY o.cedis, o.pais
        ORDER BY total_sustituciones DESC
        LIMIT 15;
    """, conn)
    
    report_content += "## 3. CEDIS con Mayor Número de Sustituciones (Top 15)\n"
    report_content += top_cedis.to_markdown(index=False) + "\n\n"
    
    # 5. Most affected customers
    print("Calculando clientes más afectados...")
    top_customers = pd.read_sql_query("""
        SELECT 
            o.customer_id,
            o.pais,
            o.business_unit,
            count(s.id_linea) as total_sustituciones
        FROM orders o
        JOIN sustituciones s ON o.id_pedido = s.id_pedido
        GROUP BY o.customer_id, o.pais, o.business_unit
        ORDER BY total_sustituciones DESC
        LIMIT 15;
    """, conn)
    
    report_content += "## 4. Clientes Más Afectados por Sustituciones (Top 15)\n"
    report_content += top_customers.to_markdown(index=False) + "\n\n"
    
    # 6. Analysis by Business Unit
    print("Calculando sustituciones por Unidad de Negocio...")
    bu_sub = pd.read_sql_query("""
        SELECT 
            o.business_unit,
            count(distinct o.id_pedido) as total_pedidos_bu,
            count(distinct s.id_pedido) as pedidos_con_sustitucion,
            count(s.id_linea) as total_sustituciones
        FROM orders o
        LEFT JOIN sustituciones s ON o.id_pedido = s.id_pedido
        GROUP BY o.business_unit;
    """, conn)
    
    report_content += "## 5. Sustituciones por Unidad de Negocio (Business Unit)\n"
    report_content += bu_sub.to_markdown(index=False) + "\n\n"
    
    # Write report
    report_path = os.path.join(base_dir, "reporte_analisis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"Reporte de análisis guardado en {report_path}!")
    conn.close()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_analysis(current_dir)
