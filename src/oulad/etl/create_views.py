import os
import psycopg2
from psycopg2 import sql
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

console = Console()

def run_sql_script(conn, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        script = f.read()

    with conn.cursor() as cur:
        cur.execute(script)
        conn.commit()

def main():
    # Configura tus credenciales
    conn_info = {
        "dbname": "oulad_db",
        "user": "oulad_user",
        "password": "oulad_pass",
        "host": "localhost",
        "port": "5432"
    }

    # Rutas de los scripts
    base_path = "sql/full_domain"
    scripts = [
        "fd_student_vles.sql",           # vista normal
        "fd_student_assessments.sql"    # vista materializada
    ]

    console.print(Panel("[bold cyan]Creación de Vistas Full Domain[/bold cyan]", expand=False))

    try:
        with psycopg2.connect(**conn_info) as conn:
            with Progress() as progress:
                task = progress.add_task("[green]Ejecutando scripts...", total=len(scripts))

                for script in scripts:
                    file_path = os.path.join(base_path, script)
                    run_sql_script(conn, file_path)
                    progress.advance(task)
                    console.print(f"[bold green]✔[/bold green] Ejecutado: [yellow]{script}[/yellow]")

        console.print(Panel("[bold green]Vistas creadas exitosamente.[/bold green]", expand=False))

    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red] {str(e)}", expand=False))

if __name__ == "__main__":
    main()