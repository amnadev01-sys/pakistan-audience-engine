"""Stage 01 (inspect): print candidate video titles for a manual quality check.

Read-only: this script does not change any file.
"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

candidates = pd.read_csv("data/raw/candidates.csv")

print(f"Candidate videos: {len(candidates)}\n")
for index, row in candidates.iterrows():
    status = "PASS" if row["quality_pass"] else "fail"
    print(f"{index:>2} | {status} | {row['channel'][:18]:<18} | "
          f"{row['content_type']:<12} | {row['title'][:85]}")