from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
import cv2
import numpy as np


class ParagraphCropper:

    def __init__(self):
        filepath = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        self.model = YOLOv10(filepath)

    def crop_paragraphs(self, image: np.ndarray, threshold: float = 0.9):

        # Perform layout parsing
        results = self.model(
            image,
            conf=0.2,  # Confidence threshold
            device="cuda",
        )  # Adjust confidence threshold as needed

        boxes = results[0].boxes.xyxy
        labels = results[0].boxes.cls
        conf = results[0].boxes.conf
        # Extract and save paragraphs
        required_boxes = boxes[(labels == 0) | (labels == 1)]
        required_boxes = required_boxes[conf[(labels == 0) | (labels == 1)] > threshold]  # 
        
        crops = []
        for box in required_boxes:
            x1, y1, x2, y2 = map(int, box)
            paragraph_crop = image[y1:y2, x1:x2]
            crops.append(paragraph_crop)
            # cv2.imwrite(f"paragraph_{x1}_{y1}.jpg", paragraph_crop)
            # print(f"Cropped paragraph saved as paragraph_{x1}_{y1}.jpg")

        return crops
