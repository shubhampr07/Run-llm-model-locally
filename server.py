from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)

MODEL_PATH = "C:/Users/Shubham Kr Mandal/Desktop/llama-1b/LaMini-Flan-T5-248M"

print("🚀 Loading model from:", MODEL_PATH)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")

    print(f"📩 Received prompt: {prompt}") 

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"📤 Generated response: {generated_text}")
    return jsonify({"response": generated_text})

if __name__ == "__main__":
    print("🌍 Server starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)



#### Use this code if you are using causal model or new models####


# from flask import Flask, request, jsonify
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# app = Flask(__name__)

# # Load your model (Replace with your model's path if downloaded)
# MODEL_PATH = "C:/Users/Shubham Kr Mandal/Desktop/llama-1b/LaMini-Flan-T5-248M"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).eval()

# @app.route("/generate", methods=["POST"])
# def generate_text():
#     data = request.json
#     prompt = data.get("prompt", "")
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=100)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return jsonify({"response": generated_text})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
