import json
import sys


notebook = json.load(sys.stdin)
for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

json.dump(notebook, sys.stdout, ensure_ascii=False, indent=1)
sys.stdout.write("\n")
