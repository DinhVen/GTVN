"""
Sửa mã nhóm R.122 -> P.122 trong metadata (STOP là biển cấm theo QCVN 41)
Chỉ sửa cột label, KHÔNG sửa image_path -> không cần rebuild FAISS
"""
import pandas as pd

METADATA_PATH = "d:/gtvn/data/metadata.csv"
df = pd.read_csv(METADATA_PATH)

# Count trước khi sửa
r122_count = len(df[df['label'] == 'R.122'])
print(f"Found {r122_count} rows with label R.122")

# Đổi label R.122 -> P.122
df.loc[df['label'] == 'R.122', 'label'] = 'P.122'

# Cập nhật meaning và advice cho P.122
df.loc[df['label'] == 'P.122', 'meaning'] = 'Dừng lại (STOP)'
df.loc[df['label'] == 'P.122', 'advice'] = 'Bắt buộc dừng xe hoàn toàn trước vạch dừng, quan sát an toàn rồi mới tiếp tục đi.'

# Verify
p122_count = len(df[df['label'] == 'P.122'])
print(f"After fix: {p122_count} rows with label P.122")
print(f"R.122 remaining: {len(df[df['label'] == 'R.122'])}")

# Thống kê nhóm mới
labels = sorted(df['label'].unique())
groups = {}
for l in labels:
    prefix = l.split('.')[0]
    groups[prefix] = groups.get(prefix, 0) + 1
print(f"\nTotal unique labels: {len(labels)}")
for g, c in sorted(groups.items()):
    print(f"  {g}: {c} labels")

df.to_csv(METADATA_PATH, index=False)
print("\nDone! Saved metadata.csv")
