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


def _is_gpu_env_active() -> bool:
    try:
        prefix = os.path.normcase(sys.prefix)
        return os.path.normcase(os.path.join("envs", "gpu")) in prefix
    except Exception:
        return False


def _find_conda_exe() -> str | None:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and os.path.exists(conda_exe):
        return conda_exe

    # Common default for this workspace/user
    candidate = os.path.join(os.path.expanduser("~"), "anaconda3", "Scripts", "conda.exe")
    if os.path.exists(candidate):
        return candidate

    return None

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    app_path = os.path.join(PROJECT_DIR, "app.py")

    # If the gpu env is not active, prefer launching via `conda run -n gpu`
    # so imports like `ultralytics` work consistently.
    if not _is_gpu_env_active():
        conda_exe = _find_conda_exe()
        if conda_exe:
            cmd = [
                conda_exe,
                "run",
                "-n",
                "gpu",
                "--no-capture-output",
                "python",
                "-m",
                "streamlit",
                "run",
                app_path,
            ]
        else:
            cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]

    # Append any extra CLI args the user passed
    if len(sys.argv) > 1:
        cmd += sys.argv[1:]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("Streamlit exited with error:", e)
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()
