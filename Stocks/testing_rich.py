from rich.text import Text
from rich.console import Console

console = Console()

# Styles and Markup
console.print("[bold]Bold[/bold], [italic]Italic[/], [underline]Underline[/]")
console.print("[strike]Strikethrough[/], [dim]Dim text[/]")
console.print("[reverse]Reverse colors[/] and [blink]Blinking[/] text")

# Colors and Backgrounds
console.print("[red]Red[/] on [green]Green[/] background")
console.print("[white on blue]White on blue background[/]")

# Hex and RGB colors
console.print("Hex color", style="#ff00ff")         # Magenta text
console.print("RGB color", style="rgb(128,128,255)") # Pastel blue text

# Alignment and Overflow
console.print("Centered Text", style="on blue", justify="center", width=40)
console.print("Too long to show completely", overflow="ellipsis", width=18)

# Rainbow text effect example:
text = Text()
colors = ["red", "yellow", "green", "cyan", "blue", "magenta"]
for idx, char in enumerate("RAINBOW!"):
    text.append(char, style=colors[idx % len(colors)])
console.print(text)

from rich.progress import Progress
import time

def process_items(items):
    # Dummy fast function (simulated work)
    time.sleep(0.01)

items = range(100)
with Progress(refresh_per_second=10) as progress:  # refresh ~10 times/sec
    task = progress.add_task("Processing...", total=len(items))
    for item in items:
        process_items([item])
        progress.update(task, advance=1)