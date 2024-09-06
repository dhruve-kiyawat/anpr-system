from fastapi import APIRouter, UploadFile, File
from ..models.RestfulModel import RestfulModel


from ..utils.RouterFunctions import yolo_predict, ocr_predict, anpr_predict

router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post("/yolo-predict/", summary="Perform Number Plate Detection on uploded image")
async def YoloPredict(file: UploadFile = File(...)):
    return await yolo_predict(file)

@router.post('/ocr-predict', response_model=RestfulModel, summary="Perform Optical Character Recognition on uploded image")
async def OcrPredict(file: UploadFile = File(...)):
    return ocr_predict(file)

@router.post('/anpr-predict', summary="Perform Number Plate Recognition on uploded image")
async def Predict(file: UploadFile = File(...)):
    return await anpr_predict(file)