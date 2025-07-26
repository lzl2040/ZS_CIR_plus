import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('/data/tangwenyue/Code/ZS-CIR/ZS-CIR-twy/data/nli_for_simcse.csv')

# 保存为 XLSX 文件
df.to_excel('/data/tangwenyue/Code/ZS-CIR/ZS-CIR-twy/data/nli_for_simcse.xlsx', index=False)  # index=False 可选，表示不保存行号
print("Successfully convert the file!")