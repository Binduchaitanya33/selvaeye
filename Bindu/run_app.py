"""
Entry point wrapper to launch the Streamlit `app.py` from an executable.
This script forwards arguments to `streamlit run app.py` and opens the
Streamlit server in the default browser.

Note: Packaging Streamlit apps into a single EXE can be fragile; if the
PyInstaller build fails, follow the README instructions to run the app
with Python and Streamlit instead.
"""
import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # Ensure we run streamlit from the project directory
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.join(PROJECT_DIR, "app.py")]
    # Append any extra CLI args the user passed to the EXE
    if len(sys.argv) > 1:
        cmd += sys.argv[1:]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("Streamlit exited with error:", e)
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()
