import pandas as pd

v1 = pd.read_csv('d:/gtvn/data/metadata_v1.csv')
cur = pd.read_csv('d:/gtvn/data/metadata.csv')

labels = sorted(v1['label'].unique())

print("=== LABELS WHERE ORIGINAL HAD MEANING, BUT WE POTENTIALLY OVERWROTE ===")
count = 0
for lbl in labels:
    v1_rows = v1[v1['label'] == lbl]
    cur_rows = cur[cur['label'] == lbl]
    if len(v1_rows) == 0 or len(cur_rows) == 0:
        continue
    
    v1m = str(v1_rows.iloc[0].get('meaning', '')).strip()
    cm = str(cur_rows.iloc[0].get('meaning', '')).strip()
    v1a = str(v1_rows.iloc[0].get('advice', '')).strip()
    ca = str(cur_rows.iloc[0].get('advice', '')).strip()
    
    has_v1_meaning = v1m != '' and v1m != 'nan'
    diff_meaning = v1m != cm
    
    if has_v1_meaning and diff_meaning:
        count += 1
        if count <= 40:
            print(f"\n{lbl}:")
            print(f"  ORIG meaning: {v1m[:80]}")
            print(f"  NEW meaning:  {cm[:80]}")
            print(f"  ORIG advice:  {v1a[:80]}")
            print(f"  NEW advice:   {ca[:80]}")

print(f"\n\nTotal: {count} labels had original meanings that were overwritten")
print(f"Total labels with EMPTY meaning in v1: {len([l for l in labels if str(v1[v1['label']==l].iloc[0].get('meaning','')).strip() in ('','nan')])}")
