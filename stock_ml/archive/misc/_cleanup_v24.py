"""Remove dead code from run_v24.py after second DONE"""
import os
path = os.path.join(os.path.dirname(__file__), "run_v24.py")
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the second occurrence of 'print("DONE")'
done_count = 0
cut_line = len(lines)
for i, line in enumerate(lines):
    if 'print("DONE")' in line:
        done_count += 1
        if done_count == 2:
            # Keep up to the line after print("=" * 130) following DONE
            cut_line = i + 2  # DONE line + "=" line + 1
            break

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines[:cut_line])

print(f"Kept {cut_line} of {len(lines)} lines. Removed {len(lines) - cut_line} dead lines.")
