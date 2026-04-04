import streamlit as st
import os
from PIL import Image
import shutil
import subprocess

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(PROJECT_ROOT, 'inputs')
OUTPUT_IMAGE = os.path.join(PROJECT_ROOT, 'outputs', 'final', 'tryon_with_background.png')
PIPELINE_SCRIPT = os.path.join(PROJECT_ROOT, 'src', 'master_pipeline.py')


st.markdown("""
<h1 style='text-align: center; color: #4F8BF9;'>Virtual Try-On</h1>
<p style='text-align: center;'>Upload a person image and a garment image. The pipeline will run and display the try-on result.</p>
""", unsafe_allow_html=True)

# File uploaders side by side
col1, col2 = st.columns(2)
with col1:
    person_file = st.file_uploader('Person Image', type=['jpg', 'jpeg', 'png', 'webp'], key='person')
    if person_file:
        person_img = Image.open(person_file)
        st.image(person_img, caption='Person', width=250)
with col2:
    garment_file = st.file_uploader('Garment Image', type=['jpg', 'jpeg', 'png', 'webp'], key='garment')
    if garment_file:
        garment_img = Image.open(garment_file)
        st.image(garment_img, caption='Garment', width=250)

run_button = st.button('Run Try-On Pipeline', use_container_width=True)

if run_button:
    if not person_file or not garment_file:
        st.error('Please upload both person and garment images.')
    else:
        os.makedirs(INPUTS_DIR, exist_ok=True)
        # Save person image (reset pointer and validate)
        person_path = os.path.join(INPUTS_DIR, 'person.jpg')
        person_file.seek(0)
        try:
            img = Image.open(person_file)
            img = img.convert('RGB')
            img.save(person_path, format='JPEG')
        except Exception as e:
            st.error(f'Person image upload is not a valid image: {e}')
            st.stop()
        # Save garment image (reset pointer and validate)
        garment_path = os.path.join(INPUTS_DIR, 'garment.jpg')
        garment_file.seek(0)
        try:
            img = Image.open(garment_file)
            img = img.convert('RGB')
            img.save(garment_path, format='JPEG')
        except Exception as e:
            st.error(f'Garment image upload is not a valid image: {e}')
            st.stop()
        st.info('Images uploaded. Running pipeline...')
        # Run pipeline
        result = subprocess.run([
            'python', PIPELINE_SCRIPT,
            '--person', person_path,
            '--garment', garment_path
        ], capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            st.error('Pipeline failed!')
            with st.expander('Show pipeline output'):
                st.text(result.stdout)
                st.text(result.stderr)
        else:
            st.success('Pipeline completed!')
            if os.path.exists(OUTPUT_IMAGE):
                st.markdown("<h3 style='text-align: center;'>Try-On Result</h3>", unsafe_allow_html=True)
                # Resize output image for display
                output_img = Image.open(OUTPUT_IMAGE)
                max_width = 400
                w, h = output_img.size
                if w > max_width:
                    new_h = int(h * max_width / w)
                    output_img = output_img.resize((max_width, new_h))
                # Center the image using columns
                c1, c2, c3 = st.columns([1,2,1])
                with c2:
                    st.image(output_img, caption='Try-On Result', width=400)
            else:
                st.error('Output image not found!')
