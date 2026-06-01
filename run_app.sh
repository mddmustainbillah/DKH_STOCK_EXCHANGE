#!/bin/bash
# Run the IDLC DSE Streamlit dashboard using the project virtual environment.
# Usage: bash run_app.sh   (from the DKH_STOCK_EXCHANGE folder)

cd "$(dirname "$0")"
source idlc_venv/bin/activate
streamlit run app.py