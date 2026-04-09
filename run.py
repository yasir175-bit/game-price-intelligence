import subprocess
import threading
import sys
import os

def run_streamlit():
    print("Starting Streamlit Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app/streamlit_app.py"])

def run_scheduler():
    print("Starting Background Data Collector...")
    subprocess.run([sys.executable, "src/app/scheduler_runner.py"])

if __name__ == "__main__":
    # Create thread for scheduler
    t1 = threading.Thread(target=run_scheduler)
    t1.daemon = True
    t1.start()

    # Run Streamlit on main thread
    run_streamlit()
