import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# モデルとトークナイザーのロード
model_name = "EleutherAI/gpt-neox-20b"  # 使用するモデル名
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# JSONファイルの読み込み
with open("embedding_data.json", "r") as f:
    data = json.load(f)

# 埋め込みの生成
embeddings = {}
for key, text in data.items():
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings[key] = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# gguf形式に変換して保存（仮想的な例）
# import gguf  # 仮のモジュール名
# gguf_data = gguf.convert_from_numpy(embeddings)
# gguf.save("embeddings.gguf", gguf_data)

# 実際の保存
np.save("embeddings.npy", embeddings)  # 仮の保存方法
