import pandas as pd
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('d:/gtvn/data/metadata.csv')
labels = sorted(df['label'].unique())

print(f"Total unique labels: {len(labels)}")
print("\nAll labels with current meaning:")
for lbl in labels:
    row = df[df['label'] == lbl].iloc[0]
    m = str(row.get('meaning', '')).strip()
    a = str(row.get('advice', '')).strip()
    # Flag suspicious ones
    flag = ""
    if m == '' or m == 'nan':
        flag = " *** EMPTY ***"
    elif 'tốc độ' in m.lower() and lbl.startswith('P.124'):
        flag = " *** SUSPICIOUS - P.124 should be quay đầu ***"
    print(f"  {lbl}: meaning=[{m[:70]}]{flag}")
