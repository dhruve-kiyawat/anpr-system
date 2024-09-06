import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.markdown("""
    <style>
    .button-container {
        display: flex;
        justify-content: center;
        gap: 10px;
    }
    .button-container > div {
        flex-grow: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to upload file and call FastAPI endpoint
def ocr_predict(file):
    url = f'http://localhost:{8000}/ocr/ocr-predict'  # Replace with your FastAPI endpoint URL
    files = {'file': ('image.jpg', file, 'image/jpeg')}
    response = requests.post(url, files=files)
    return response.json()

def yolo_predict(file):
    url = f'http://localhost:{8000}/ocr/yolo-predict'  # Replace with your FastAPI endpoint URL
    files = {'file': ('image.jpg', file, 'image/jpeg')}
    response = requests.post(url, files=files)
    return response

def anpr_predict(file):
    url = f'http://localhost:{8000}/ocr/anpr-predict'  # Replace with your FastAPI endpoint URL
    files = {'file': ('image.jpg', file, 'image/jpeg')}
    response = requests.post(url, files=files)
    return response

BUTTON_PRESSED = 0

# Streamlit app
def main():
    global BUTTON_PRESSED
    st.title('Upload Image for OCR Prediction')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)  # Create 3 columns for the buttons
        result_image_placeholder = st.empty()
        result_text_placeholder = st.empty()

        with col1:
            if st.button('OCR Predict'):
                BUTTON_PRESSED = 1
                result_text = action(uploaded_file)
                result_text_placeholder.text(result_text)


        with col2:
            if st.button('YOLO Predict'):
                BUTTON_PRESSED = 2
                result_image = action(uploaded_file)
                result_image_placeholder.image(result_image, caption='Result Image', use_column_width=True)


        with col3:
            if st.button('ANPR Predict'):
                BUTTON_PRESSED = 3
                result_dict = action(uploaded_file)
                result_image_placeholder.image(result_dict['result_image'], caption='Result Image', use_column_width=True)
                result_text_placeholder.text(str(result_dict['result_text']))
        
                

def action(uploaded_file):
    global BUTTON_PRESSED
    result = "No result"

    try:
        image_bytes = uploaded_file.read() # Convert uploaded file to bytes


        if BUTTON_PRESSED == 1:
            response = ocr_predict(image_bytes) # Call FastAPI endpoint
            result = [] if response['data'][0] is None else [i[1][0] for i in response['data'][0]]
            result = str(result)
            print(BUTTON_PRESSED)
           

        elif BUTTON_PRESSED == 2:
            print(BUTTON_PRESSED)
            response = yolo_predict(image_bytes) # Call FastAPI endpoint
            image_bytes = response.content
            image = Image.open(BytesIO(image_bytes))
            result = image
            # print(len(result))


        elif BUTTON_PRESSED == 3:
            print(BUTTON_PRESSED)
            response = anpr_predict(image_bytes) # Call FastAPI endpoint
            image_bytes = response.content
            image = Image.open(BytesIO(image_bytes))
            ocr_result = response.headers.get('ocr-results')
            result = {'result_image' : image, 
                      'result_text' : ocr_result}
            

    except Exception as e:
        result = f'Error: {e}'

    return result

if __name__ == "__main__":
    main()