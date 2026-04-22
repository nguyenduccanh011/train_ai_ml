"""Generator script to create run_v24.py"""
import shutil, os

SRC = os.path.join(os.path.dirname(__file__), "run_v23_optimal.py")
DST = os.path.join(os.path.dirname(__file__), "run_v24.py")

with open(SRC, "r", encoding="utf-8") as f:
    code = f.read()

print("Copied V23 -> V24 base, length:", len(code))
# We will manually edit the file after copying
shutil.copy2(SRC, DST)
print("Done. Now edit run_v24.py with patches.")
