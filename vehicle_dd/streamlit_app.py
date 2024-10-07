import streamlit as st
import requests
from PIL import Image
import io

# Streamlit UI
st.title("Car Damage Detection")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes for API request
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Options for user: Predict or Predict and Visualize
    option = st.radio("Choose an option:", ("Predict", "Predict and Visualize"))

    # Predict button
    if st.button("Submit"):
        if option == "Predict":
            try:
                # Send the image to FastAPI backend for prediction
                files = {"file": ("filename", img_bytes, "image/jpeg")}
                response = requests.post("http://127.0.0.1:8000/predict", files=files)

                if response.status_code == 200:
                    predictions = response.json()
                    st.write("Predictions:")
                    st.json(predictions)  # Display predictions as JSON
                else:
                    st.write(f"Error {response.status_code}: {response.text}")

            except Exception as e:
                st.write(f"Error occurred: {e}")

        elif option == "Predict and Visualize":
            try:
                # Send the image to FastAPI backend for visualization
                files = {"file": ("filename", img_bytes, "image/jpeg")}
                response = requests.post("http://127.0.0.1:8000/predict_and_visualize", files=files)

                if response.status_code == 200:
                    # Display the visualized image
                    result_image = Image.open(io.BytesIO(response.content))
                    st.image(result_image, caption="Predicted Image", use_column_width=True)
                else:
                    st.write(f"Error {response.status_code}: {response.text}")

            except Exception as e:
                st.write(f"Error occurred: {e}")
