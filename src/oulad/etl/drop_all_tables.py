from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

DATABASE_URL = "postgresql://oulad_user:oulad_pass@localhost:5432/oulad_db"
engine = create_engine(DATABASE_URL)

console = Console()

def drop_all_tables(engine):
    drop_statements = [
        "DROP TABLE IF EXISTS student_vles_raw CASCADE;",
        "DROP TABLE IF EXISTS vles_raw CASCADE;",
        "DROP TABLE IF EXISTS student_assessments_raw CASCADE;",
        "DROP TABLE IF EXISTS assessments_raw CASCADE;",
        "DROP TABLE IF EXISTS student_registrations_raw CASCADE;",
        "DROP TABLE IF EXISTS student_infos_raw CASCADE;",
        "DROP TABLE IF EXISTS courses_raw CASCADE;",

        "DROP TABLE IF EXISTS student_vles CASCADE;",
        "DROP TABLE IF EXISTS vles CASCADE;",
        "DROP TABLE IF EXISTS student_assessments CASCADE;",
        "DROP TABLE IF EXISTS assessments CASCADE;",
        "DROP TABLE IF EXISTS student_registrations CASCADE;",
        "DROP TABLE IF EXISTS student_infos CASCADE;",
        "DROP TABLE IF EXISTS courses CASCADE;",

        "DROP VIEW IF EXISTS fd_student_vle;",
        "DROP MATERIALIZED VIEW IF EXISTS fd_student_assessments;"
    ]
    console.rule("[bold red]🗑️ Eliminando Tablas y Vistas")

    try:
        with engine.connect() as conn:
            for stmt in drop_statements:
                console.print(f"[cyan]Ejecutando:[/cyan] {stmt.strip()}")
                conn.execute(text(stmt))
            conn.commit()
        console.print(Panel.fit("[green]Todas las tablas RAW y CLEAN fueron eliminadas correctamente.[/green]", title="Éxito", style="bold green"))

    except SQLAlchemyError as e:
        console.print(Panel.fit(f"[red]Error al eliminar las tablas:[/red]\n\n{str(e)}", title="Error", style="bold red"))

if __name__ == "__main__":
    drop_all_tables(engine)
