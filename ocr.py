import cv2
import pytesseract
import numpy as np

def perform_multilingual_ocr(crops: list[np.ndarray], languages: list[str] = ['eng']) -> list[str]:
    """
    Takes a list of cropped image arrays and extracts text using multiple languages.
    
    Args:
        crops: List of BGR image arrays from OpenCV.
        languages: List of Tesseract language codes (e.g., ['guj', 'eng', 'hin']).
    """
    extracted_texts = []
    
    # Join the language list with a '+' as Tesseract expects (e.g., "guj+eng+hin")
    lang_string = "+".join(languages)
    print(f"Running OCR with languages: {lang_string}")
    
    # PSM 6: Assume a single uniform block of text.
    # OEM 3: Default (uses LSTM neural net)
    custom_config = r'--oem 3 --psm 6'
    
    for i, crop_img in enumerate(crops):
        if crop_img.size == 0:
            extracted_texts.append("")
            continue
            
        # --- Preprocessing ---
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Otsu's thresholding for clean black/white text
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # --- OCR Extraction ---
        try:
            # Pass the combined language string here
            text = pytesseract.image_to_string(
                thresh, 
                lang=lang_string, 
                config=custom_config
            )
            
            extracted_texts.append(text.strip())
            
        except pytesseract.TesseractError as e:
            print(f"Tesseract failed on crop {i}: {e}")
            extracted_texts.append("")
            
    return extracted_texts