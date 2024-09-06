from fastapi import HTTPException, UploadFile, status, File
from fastapi.responses import StreamingResponse
from ..models.RestfulModel import RestfulModel

import numpy as np
import cv2
from PIL import Image
from io import BytesIO

from paddleocr import PaddleOCR
from ultralytics import YOLO

ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
yolo_model = YOLO(r'app\backend\pretrained_yolo\best.pt')


def ocr_predict(file: UploadFile = File(...)):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        img = np.array(Image.open(file.file))
        result = ocr_model.ocr(img=img, cls=True)
        restfulModel.data = result
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a .jpg or .png format image"
        )
    return restfulModel


async def yolo_predict(file: UploadFile = File(...)):
    if file.filename.endswith((".jpg", ".png")):
        # Read the uploaded file into memory
        image = Image.open(BytesIO(await file.read()))

        # YOLO object detection, bounding box extraction, and plotting on image.
        result = yolo_model.predict(source = image)[0]
        pred_bounding_box = result.boxes.xyxy.cpu().numpy()
        plotted_image, _ =  plot_bboxes(image, pred_bounding_box, color=(255, 0, 0), mode = "Predicted")

        # Save the image to a BytesIO object
        img_byte_arr = BytesIO()
        plotted_image.save(img_byte_arr, format=image.format)
        img_byte_arr.seek(0)
        
        # Determine media format and return the image as a StreamingResponse
        media_type = f"image/{image.format.lower()}"
        
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a .jpg or .png format image"
        )
    
    return StreamingResponse(img_byte_arr, media_type=media_type)


async def anpr_predict(file: UploadFile = File(...)):
    if file.filename.endswith((".jpg", ".png")):
        # Read the uploaded file into memory
        image = Image.open(BytesIO(await file.read()))

        # YOLO object detection, bounding box extraction, and plotting on image.
        result = yolo_model.predict(source = image)[0]
        pred_bounding_box = result.boxes.xyxy.cpu().numpy()
        plotted_image, ocr_results =  plot_bboxes(image, pred_bounding_box, color=(255, 0, 0), mode = "OCR")

        # Save the image to a BytesIO object
        img_byte_arr = BytesIO()
        plotted_image.save(img_byte_arr, format=image.format)
        img_byte_arr.seek(0)
        
        # Determine media format and return the image as a StreamingResponse
        media_type = f"image/{image.format.lower()}"
        
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a .jpg or .png format image"
        )
    
    headers = {"OCR-Results": str(ocr_results)}
    return StreamingResponse(img_byte_arr, media_type=media_type, headers=headers)


def plot_bboxes(image, bboxes, color=(255, 0, 0), mode="OCR", thickness=2):
    # Convert PIL image to OpenCV image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    ocr_results = []
    
    if len(bboxes) > 0:
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = map(int, bbox)
            if mode == "OCR":
                bbox_image =  image[ymin:ymax, xmin:xmax]
        
                # Perform OCR on the cropped bounding box
                result = ocr_model.ocr(img= bbox_image, cls=True)
                result = [] if result[0] is None else [i[1][0] for i in result[0]]
                ocr_results.append(result)
            
            # Draw solid background for the 
            text_size, _ = cv2.getTextSize("Predicted", cv2.FONT_HERSHEY_SIMPLEX, 1, thickness)
            cv2.rectangle(image, (xmin, ymin - text_size[1] - 10), (xmin + text_size[0], ymin - 10), color, -1)
            
            # Draw bounding box and text
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
            cv2.putText(image, "Predicted", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness, cv2.LINE_AA)
    
    else:
        # Draw solid background for "No Detection" message
        text_size, _ = cv2.getTextSize("No Detection", cv2.FONT_HERSHEY_SIMPLEX, 1, thickness)
        cv2.rectangle(image, (0, 0), (20 + text_size[0], 20 + text_size[1]), color, -1)
        
        # Draw "No Detection" message
        cv2.putText(image, "No Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness, cv2.LINE_AA)

    # Convert OpenCV image back to PIL image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image, ocr_results