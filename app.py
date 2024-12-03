from flask import Flask, request, jsonify, send_from_directory
from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image
import uuid
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Paths
GENERATED_IMAGES_PATH = "./generated_images"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
app.config['UPLOAD_FOLDER'] = GENERATED_IMAGES_PATH

# Load the pre-trained model
print("Loading the model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Load the pipeline from Hugging Face
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
    pipe = pipe.to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None

# Ensure the generated images directory exists
os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)

@app.route('/generate', methods=['POST'])
def generate_images():
    if pipe is None:
        return jsonify({"error": "Model not loaded. Please check server logs for details."}), 500

    try:
        # Parse the request data
        data = request.json
        prompt = data.get("prompt", "")
        num_images = int(data.get("num_images", 1))

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        if num_images < 1 or num_images > 10:
            return jsonify({"error": "num_images should be between 1 and 10"}), 400

        # Generate images
        images = []
        for _ in range(num_images):
            image = pipe(prompt).images[0]
            filename = f"{uuid.uuid4()}.png"
            filepath = os.path.join(GENERATED_IMAGES_PATH, filename)
            image.save(filepath)
            images.append(f"/images/{filename}")  # Adjusted path for serving

        return jsonify({"images": images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/images/<filename>', methods=['GET'])
def serve_image(filename):
    """
    Serve generated images from the `generated_images` directory.
    """
    try:
        return send_from_directory(GENERATED_IMAGES_PATH, filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
