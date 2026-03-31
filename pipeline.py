"""
AksharAI Document Extraction Pipeline (with Hugging Face Hub Sync)

This module orchestrates the end-to-end process of converting scanned documents (PDFs)
into structured, machine-readable text. It exports the results as a Hugging Face Dataset
and safely concatenates the new data with existing data on the HF Hub.
"""

import os
import cv2
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Hugging Face Datasets
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub.utils import RepositoryNotFoundError

# Import functionality from your existing modules
from pdf_to_image import split_pdf_to_images
from paragraph_crop import ParagraphCropper
from ocr import perform_multilingual_ocr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    A scalable pipeline to extract text from PDFs and sync to Hugging Face Datasets.
    """

    def __init__(self, languages: List[str] = ['guj', 'eng'], max_ocr_workers: int = 4):
        logger.info("Initializing DocumentPipeline...")
        self.languages = languages
        self.max_ocr_workers = max_ocr_workers
        
        logger.info("Loading YOLOv10 Layout Model...")
        self.cropper = ParagraphCropper()
        logger.info("Pipeline initialized successfully.")

    def process_pdf(self, pdf_path: str, output_image_dir: str = "temp_images", dpi: int = 300) -> List[Dict[str, Any]]:
        """Executes the full extraction pipeline on a given PDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Source PDF not found: {pdf_path}")

        logger.info(f"Starting pipeline for PDF: {pdf_path}")
        document_results = []

        try:
            # Step 1: Convert PDF to High-Res Images
            page_image_paths = split_pdf_to_images(pdf_path, output_dir=output_image_dir, dpi=dpi)
            
            # Step 2 & 3: Layout Parsing and OCR
            for page_num, image_path in enumerate(page_image_paths):
                logger.info(f"Processing Page {page_num + 1}/{len(page_image_paths)}...")
                page_data = self._process_single_page(image_path, page_num)
                document_results.append(page_data)
                
            logger.info("Extraction execution completed.")
            return document_results

        except Exception as e:
            logger.error(f"Pipeline failed during execution: {e}")
            raise

    def _process_single_page(self, image_path: str, page_num: int) -> Dict[str, Any]:
        """Processes a single image page for paragraphs and OCR."""
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to read image at {image_path}. Skipping page.")
            return {"page": page_num + 1, "paragraphs": []}

        crops = self.cropper.crop_paragraphs(image, threshold=0.85)
        logger.info(f"Page {page_num + 1}: Detected {len(crops)} paragraphs.")

        extracted_texts = self._parallel_ocr(crops)

        return {
            "page": page_num + 1,
            "paragraphs": extracted_texts
        }

    def _parallel_ocr(self, crops: List['np.ndarray']) -> List[str]:
        """Executes OCR on multiple image crops concurrently."""
        results: List[Optional[str]] = [None] * len(crops)
        
        def _ocr_single_crop(index: int, crop: 'np.ndarray') -> tuple:
            text_list = perform_multilingual_ocr([crop], languages=self.languages)
            return index, text_list[0] if text_list else ""

        with ThreadPoolExecutor(max_workers=self.max_ocr_workers) as executor:
            future_to_index = {
                executor.submit(_ocr_single_crop, i, crop): i 
                for i, crop in enumerate(crops)
            }
            for future in as_completed(future_to_index):
                try:
                    index, text = future.result()
                    results[index] = text
                except Exception as exc:
                    index = future_to_index[future]
                    logger.error(f"OCR thread failed for crop {index}: {exc}")
                    results[index] = "" 

        return [text for text in results if text is not None]

    def export_to_hf_dataset(self, document_results: List[Dict[str, Any]], source_name: str = "unknown_pdf") -> Dataset:
        """Converts extracted text into a Hugging Face Dataset."""
        logger.info("Converting extracted data to Hugging Face Dataset format...")
        
        hf_data_dict = {
            "source_document": [],
            "page_number": [],
            "paragraph_index": [],
            "text": []
        }
        
        for page_data in document_results:
            page_num = page_data["page"]
            for p_idx, para_text in enumerate(page_data["paragraphs"]):
                if not para_text.strip():
                    continue
                    
                hf_data_dict["source_document"].append(source_name)
                hf_data_dict["page_number"].append(page_num)
                hf_data_dict["paragraph_index"].append(p_idx)
                hf_data_dict["text"].append(para_text)
                
        dataset = Dataset.from_dict(hf_data_dict)
        logger.info(f"Created local dataset with {len(dataset)} valid paragraphs.")
        return dataset

    def push_and_concatenate_to_hub(self, new_dataset: Dataset, repo_id: str, hf_token: Optional[str] = None):
        """
        Pushes a new dataset to the Hugging Face Hub. If a dataset already exists at
        the given repo_id, it downloads it, concatenates the new data, and pushes the updated version.
        
        Args:
            new_dataset (Dataset): The newly generated Hugging Face Dataset.
            repo_id (str): The target repository on HF Hub (e.g., 'neetmehta/akshar-ai-corpus').
            hf_token (str, optional): Your HF write token. If None, it assumes you are logged in via CLI.
        """
        logger.info(f"Preparing to upload dataset to Hugging Face Hub at: {repo_id}")
        
        try:
            # Attempt to load the existing dataset from the Hub (defaulting to the 'train' split)
            logger.info("Checking for existing dataset on the Hub...")
            existing_dataset = load_dataset(repo_id, split="train", token=hf_token)
            
            logger.info(f"Found existing dataset with {len(existing_dataset)} rows. Merging with new data...")
            
            # Ensure the schemas match before concatenating.
            # cast_column can be used here if schemas mismatch slightly, but for this pipeline they are identical.
            final_dataset = concatenate_datasets([existing_dataset, new_dataset])
            
        except (RepositoryNotFoundError, ValueError, FileNotFoundError) as e:
            # If the repo doesn't exist, is empty, or there's no data to load, we push as a fresh dataset.
            logger.info("No existing dataset found (or repository is empty). Initializing as a new dataset.")
            final_dataset = new_dataset
            
        except Exception as e:
            # Catch unexpected errors (e.g., network timeout)
            logger.error(f"Failed to load existing dataset due to an unexpected error: {e}")
            raise

        # Push the combined (or new) dataset back to the Hub
        logger.info(f"Pushing dataset with {len(final_dataset)} total rows to the Hub...")
        final_dataset.push_to_hub(repo_id, token=hf_token)
        logger.info("Successfully pushed dataset to the Hugging Face Hub!")


if __name__ == "__main__":
    # --- Configuration ---
    PDF_FILE = r"F:\projects\akshar-ai\data\botanycompleteco00indr_bw-521.pdf"
    HF_REPO_ID = "neetmehta/akshar-ai-gujarati-corpus" # Replace with your target repo
    HF_TOKEN = None # Or insert your specific "hf_..." token string here
    
    pipeline = DocumentPipeline(languages=['guj'], max_ocr_workers=4)
    
    try:
        # 1. Run the extraction pipeline
        extracted_document = pipeline.process_pdf(
            pdf_path=PDF_FILE, 
            output_image_dir="processed_pages", 
            dpi=300
        )
        
        # 2. Convert to Hugging Face Dataset
        hf_dataset = pipeline.export_to_hf_dataset(
            document_results=extracted_document, 
            source_name=PDF_FILE
        )
        
        # 3. Concatenate (if applicable) and Push to Hub
        if len(hf_dataset) > 0:
            pipeline.push_and_concatenate_to_hub(
                new_dataset=hf_dataset, 
                repo_id=HF_REPO_ID, 
                hf_token=HF_TOKEN
            )
        else:
            logger.warning("No text extracted. Aborting Hub upload.")
                
    except FileNotFoundError:
        print(f"Please place a valid PDF named '{PDF_FILE}' in the directory to test the pipeline.")