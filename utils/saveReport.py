import pandas as pd

def save_report(report, row):
    if len(row) == 1:
        with open(report, "a", encoding="utf-8") as f:
            f.write(str(row[0]) + "\n")
    else:
        df = pd.DataFrame([row])
        df.to_csv(report, mode="a", header=False, index=False)
