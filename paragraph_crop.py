from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
import cv2
import numpy as np
import torch

class ParagraphCropper:

    def __init__(self):
        filepath = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        self.model = YOLOv10(filepath)

    def crop_paragraphs(self, image: np.ndarray, threshold: float = 0.5):

        # Perform layout parsing
        results = self.model(
            image,
            conf=0.2,  # Confidence threshold
            device="cuda",
        )  # Adjust confidence threshold as needed

        boxes = results[0].boxes.xyxy
        labels = results[0].boxes.cls
        conf = results[0].boxes.conf
        
        # 1. Define all the target labels you want to extract
        target_labels = [0, 1, 4, 6, 7, 9]
        
        # 2. Initialize an empty boolean mask on the same device as the labels tensor
        label_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # 3. Iteratively build the mask for all target labels
        for label in target_labels:
            label_mask |= (labels == label)
            
        # 4. Combine the label mask with the confidence threshold
        final_mask = label_mask & (conf > threshold)
        
        # 5. Extract the required bounding boxes
        required_boxes = boxes[final_mask]
        
        crops = []
        for box in required_boxes:
            x1, y1, x2, y2 = map(int, box)
            paragraph_crop = image[y1:y2, x1:x2]
            crops.append(paragraph_crop)
            
        annotated_frame = results[0].plot(pil=True, line_width=5, font_size=20)
        cv2.imwrite("result.jpg", annotated_frame)

        return crops
