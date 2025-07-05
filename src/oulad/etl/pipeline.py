import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

class OULADPipeline:
    def __init__(self):
        self.scripts = {
            "1": "oulad.etl.create_tables",
            "2": "oulad.etl.load_raw_tables",
            "3": "oulad.etl.transform",
            "4": "oulad.etl.create_views",
            "5": "oulad.etl.drop_all_tables"
        }

    def ejecutar(self, script_path):
        console.print(f"[cyan]🚀 Ejecutando:[/cyan] [bold]{script_path}[/bold]")
        os.environ["PYTHONPATH"] = os.path.abspath("src")
        os.system(f"python -m {script_path}")

    def mostrar_menu(self):
        console.clear()
        console.print(Panel.fit("[bold cyan]OULAD PIPELINE[/bold cyan]", border_style="bright_blue"))

        tabla = Table(show_header=True, header_style="bold magenta")
        tabla.add_column("Opción", justify="center", style="bold yellow")
        tabla.add_column("Acción", justify="left")

        tabla.add_row("1", "Crear tablas RAW y Clean")
        tabla.add_row("2", "Cargar datos RAW")
        tabla.add_row("3", "Transformar datos (imputar, codificar, features y carga)")
        tabla.add_row("4", "Crear Views (Full Domain.)")
        tabla.add_row("5", "Borrar todas las tablas (RAW y CLEAN) y Views")
        tabla.add_row("0", "Salir")

        console.print(tabla)

    def menu(self):
        while True:
            self.mostrar_menu()
            opcion = Prompt.ask("\n[bold yellow]Seleccione una opción[/bold yellow]", choices=["0", "1", "2", "3", "4", "5"])
            if opcion == "0":
                console.print("[green]👋 Saliendo del pipeline.[/green]")
                break
            elif opcion in self.scripts:
                self.ejecutar(self.scripts[opcion])
            else:
                console.print("[red]Opción inválida. Intente nuevamente.[/red]")

if __name__ == "__main__":
    pipeline = OULADPipeline()
    pipeline.menu()