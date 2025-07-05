import os
import psycopg2
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Configura tu conexión
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "oulad_db",
    "user": "oulad_user",
    "password": "oulad_pass"
}

SQL_DIRS = [Path("sql/raw_tables"), Path("sql/transformed_tables")]
console = Console()

def conectar_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        console.print("Conectado a PostgreSQL")
        return conn
    except Exception as e:
        console.print(f"Error de conexión: {e}")
        exit(1)

def ejecutar_sql(conn, sql_path):
    with open(sql_path, "r", encoding="utf-8") as file:
        sql = file.read()
    with conn.cursor() as cur:
        try:
            cur.execute(sql)
            console.print(f"Ejecutado: {sql_path.name}")
        except psycopg2.errors.DuplicateTable:
            console.print(f"Tabla ya existe: {sql_path.name}")
            conn.rollback()
        except Exception as e:
            console.print(f"Error en {sql_path.name}: {e}")
            conn.rollback()

def crear_tablas():
    conn = conectar_db()
    tabla = Table(title="Resultado de creación de tablas")
    tabla.add_column("Archivo SQL", style="cyan")
    tabla.add_column("Estado", style="green")

    for sql_dir in SQL_DIRS:
        console.print(f"\n📂 Procesando carpeta: {sql_dir}")
        sql_files = sorted(sql_dir.glob("*.sql"))
        for sql_file in sql_files:
            try:
                ejecutar_sql(conn, sql_file)
                tabla.add_row(sql_file.name, " Ejecutado")
            except Exception as e:
                tabla.add_row(sql_file.name, f"Error: {e}")

    conn.commit()
    conn.close()
    console.print(tabla)

if __name__ == "__main__":
    crear_tablas()
