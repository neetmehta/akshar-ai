import cv2
from doclayout_yolo import YOLOv10

def parse_layout(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Initialize the YOLOv10 model (make sure to specify the correct path to your model weights)
    model = YOLOv10(weights='path/to/your/yolov10_weights.pt')
    
    # Perform layout parsing
    results = model(image)
    
    # Process results (this is just an example, you may want to customize it based on your needs)
    for result in results:
        print(f"Detected {result['label']} with confidence {result['confidence']:.2f} at {result['bbox']}")
    
    return results