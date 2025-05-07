import pandas as pd

df = pd.read_json(r"D:\GithubFiles\CryptoX-for-ICML\data_construct\crypto_data\emoji_shuffle\math_500_percentage_10.jsonl", lines=True)
df = df.sample(frac=1)
for idx, item in df.iterrows():
    item = item.to_dict()
    print(f"index: {idx}\n\nanswer: {item['answer']}\n\n{item['prompt'][0]['content']}\n\n##########################################\n\n")