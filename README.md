# ncnn_paddleocr
convert paddleocr light model to ncnn,you can use it by ncnn.  
the infer code you can use chineseocr_lite project.  
PS：if you use angle model plz change the input shape dstHeight from 32 to 48  
# model support  
## text detection  
1.mv3dbnet-sim-op(paddleocr_mobile)  
2.pdocrv2.0_det-op(PP-OCRv2)  
## text angle cls  
1.angle-sim-op  
## text recognition  
1.mv3rec-sim-op(paddleocr_mobile)  
2.pdocrv2.0_rec-op(PP-OCRv2)  
![image](https://github.com/FeiGeChuanShu/ncnn_paddleocr/blob/main/ocr_app_result.jpg)
1.infer: https://github.com/DayBreak-u/chineseocr_lite/tree/onnx/cpp_projects/OcrLiteNcnn  
2.model: https://github.com/frotms/PaddleOCR2Pytorch  
