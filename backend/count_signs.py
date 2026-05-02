import pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
df = pd.read_csv('d:/gtvn/data/metadata.csv')
labels = df['label'].unique()
print(f"Total rows (images): {len(df)}")
print(f"Unique sign labels: {len(labels)}")
print()
groups = {
    'P': 'Biển cấm',
    'W': 'Biển cảnh báo',
    'R': 'Biển hiệu lệnh',
    'I': 'Biển chỉ dẫn',
    'DP': 'Biển phụ'
}
for prefix, name in groups.items():
    count = len([l for l in labels if l.startswith(prefix + '.')])
    print(f"  {prefix} - {name}: {count} loai")
