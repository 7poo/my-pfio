import shutil
import os

source_dir = r"C:\Users\HUNG\Downloads\DATA-20260123T060319Z-1-001\DATA - Copy"
dest_dir = os.getcwd()

files = ["academic_records.csv", "admission.csv", "test.csv"]

print(f"Copying files from {source_dir} to {dest_dir}...")

for f in files:
    src = os.path.join(source_dir, f)
    dst = os.path.join(dest_dir, f)
    try:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Successfully copied {f}")
        else:
            print(f"Source file not found: {src}")
    except Exception as e:
        print(f"Error copying {f}: {e}")
