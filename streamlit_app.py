import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# Initialize the model from the GitHub repository cache
@st.cache_resource
def load_model():
    # Assuming model_cache folder is uploaded to GitHub
    model_id = "CompVis/stable-diffusion-v1-4"
    cache_dir = "./model_cache"  # Use the path where the cache is saved in the GitHub repo
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Load the model
st.title("Fashion Image Generator")
pipe = load_model()

# Input for text prompt
prompt = st.text_input("Enter your text prompt (e.g., 'A stylish summer outfit')", "")

# Number of images to generate
num_images = st.slider("Number of images to generate", min_value=1, max_value=3, value=1)

# Generate images on button click
if st.button("Generate Images"):
    if prompt:
        with st.spinner("Generating images..."):
            # Generate images
            outputs = pipe([prompt] * num_images, num_inference_steps=50)
            images = outputs.images

            # Display generated images
            st.success(f"Generated {num_images} images for prompt: '{prompt}'")
            for idx, img in enumerate(images):
                st.image(img, caption=f"Image {idx + 1}", use_column_width=True)

            # Option to save images
            if st.button("Download Images"):
                for idx, img in enumerate(images):
                    img.save(f"output_{idx + 1}.png")
                st.success("Images saved successfully!")
    else:
        st.warning("Please enter a prompt to generate images.")
