# Entry point for Streamlit Cloud.
# Streamlit Cloud looks for streamlit_app.py in the repo root by default.
# This file simply re-exports the actual frontend app.

import runpy
import os
import sys

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Run the frontend app
runpy.run_path(os.path.join(ROOT, "frontend", "app.py"), run_name="__main__")
