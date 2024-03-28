import base64
import PIL
import streamlit as st
from ultralytics import YOLO

model_path = 'models/detection_model.pt'

st.set_page_config(
    page_title="Breast Mass Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styles to change the font family
font_awesome_css = """
<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.15.1/css/all.css" integrity="sha384-vp86vTRFVJgpjF9jiIGPEEqYqlDwgyBgEF109VFjmqGmIY/Y4HV4d3Gp2irVfcrp" crossorigin="anonymous">
"""
font_family = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&family=Taviraj:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
html, body, [class*="css"]  {
font-family: 'Poppins', sans-serif;
}
</style>
"""

# Add the CSS styles to the Streamlit app
st.markdown(font_awesome_css, unsafe_allow_html=True)
st.markdown(font_family, unsafe_allow_html=True)

# Display the header text at the top, bold and center-aligned
st.markdown("""
<div style="text-align: center; font-weight: bold;">
    <h4>Breast Mass Detection in Mammography Images based on Improved Deep Transformed Model</h4>
    <h5>V. RAJA SUBRAMANIAN - K. VIJAYA GOKUL</h5>
    <h5>Mentor: Dr. B. LAKSHMANAN, Associate Professor</h5>
</div>
""", unsafe_allow_html=True)
st.write("---")  # Add a horizontal line for separation
st.markdown("""
<div style="text-align: center; font-weight: bold;" > 
             <h5>Breast Mass Detection  in INbreast Dataset</h5>
             <h6>Upload an Image from INbreast dataset</h6>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Image Configuration")
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                #  use_column_width=True
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image,conf=confidence)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption='Detected Image',
                #  use_column_width=True
                 )
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")