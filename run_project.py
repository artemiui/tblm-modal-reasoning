#!/usr/bin/env python3
"""
Root-level entrypoint for Modal Logic Mechanistic Interpretability CLI Runner.
Dispatches directly to modal-logic-mi/scripts/run_project.py.
"""
import os
import sys
from pathlib import Path

# Add modal-logic-mi to sys.path
root_dir = Path(__file__).resolve().parent
modal_mi_dir = root_dir / "modal-logic-mi"
if str(modal_mi_dir) not in sys.path:
    sys.path.insert(0, str(modal_mi_dir))

# Change current working directory context if needed
from scripts.run_project import main

if __name__ == "__main__":
    main()
