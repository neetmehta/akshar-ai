import os
from pdf2image import convert_from_path

def split_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> list[str]:
    """
    Converts a scanned PDF into individual high-res images.
    Returns a list of file paths to the generated images.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    print(f"Converting {base_name}.pdf to images at {dpi} DPI...")
    # thread_count speeds up conversion for multi-page PDFs
    pages = convert_from_path(pdf_path, dpi=dpi, thread_count=16)
    
    saved_page_paths = []
    for page_num, page_img in enumerate(pages):
        # Naming convention: filename_page_0.jpg
        img_path = os.path.join(output_dir, f"{base_name}_page_{page_num}.jpg")
        page_img.save(img_path, "JPEG")
        saved_page_paths.append(img_path)
        
    print(f"Generated {len(saved_page_paths)} page images.")
    return saved_page_paths