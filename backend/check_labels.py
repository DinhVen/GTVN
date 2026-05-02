import pandas as pd
df = pd.read_csv('d:/gtvn/data/metadata.csv')
plabels = sorted([l for l in df['label'].unique() if l.startswith('P.')])
print("P labels:", plabels)
print()

# Check actual images in folders to verify P.124 variants
import os
base = 'd:/gtvn/data/dataset_aug/Prohibitory Signs'
if os.path.exists(base):
    folders = sorted(os.listdir(base))
    p124 = [f for f in folders if f.startswith('P.124')]
    print("P.124 folders:", p124)
    p125_140 = [f for f in folders if any(f.startswith(f'P.{i}') for i in range(125,141))]
    print("P.125-P.140 folders:", p125_140)
