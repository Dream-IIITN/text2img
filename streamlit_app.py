import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image

# Load the Stable Diffusion model
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# Generate images
def generate_images(prompt, num_images):
    results = pipe([prompt] * num_images, num_inference_steps=50, guidance_scale=7.5)
    return results.images

# Streamlit UI
st.title("Text-to-Image Generator")
st.markdown("Generate images based on your text descriptions!")

# Input fields
prompt = st.text_input("Enter your text prompt:", "")
num_images = st.slider("Select number of images to generate:", min_value=1, max_value=5, value=3)

if st.button("Generate"):
    if prompt.strip():
        st.markdown(f"**Generating {num_images} image(s) for:** {prompt}")
        images = generate_images(prompt, num_images)

        # Display images
        for i, img in enumerate(images):
            st.image(img, caption=f"Generated Image {i + 1}", use_column_width=True)
    else:
        st.error("Please enter a valid text prompt!")
