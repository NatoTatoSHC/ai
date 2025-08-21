from flask import Flask, request, jsonify
import torch
from ai import TinyGPT, CharTokenizer  # import your TinyGPT code

app = Flask(__name__)

# Load checkpoint
ckpt = torch.load("ckpt.pt", map_location="cpu")
itos = ckpt['itos']
stoi = ckpt['stoi']
vocab_size = len(itos)

tokenizer = CharTokenizer("")
tokenizer.itos = itos
tokenizer.stoi = stoi
tokenizer.vocab_size = vocab_size

model_cfg = ckpt['config']
model = TinyGPT(
    vocab_size=vocab_size,
    block_size=model_cfg['block_size'],
    n_layer=model_cfg['n_layer'],
    n_head=model_cfg['n_head'],
    n_embd=model_cfg['n_embd'],
    dropout=model_cfg['dropout'],
)
model.load_state_dict(ckpt['model'])
model.eval()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"response": ""})
    
    start_ids = tokenizer.encode(prompt).unsqueeze(0)
    out = model.generate(start_ids, max_new_tokens=100)
    text = tokenizer.decode(out[0].tolist())
    return jsonify({"response": text})

if __name__ == "__main__":
    app.run(debug=True)
