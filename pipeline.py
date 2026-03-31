"""
AksharAI Document Extraction Pipeline (CLI & Config Driven)

This module orchestrates the end-to-end process of converting scanned documents (PDFs)
into structured text. It reads settings from a JSON config file, securely loads the 
Hugging Face token from environment variables, and pushes the generated dataset to the Hub.
"""

import os
import cv2
import json
import logging
import argparse
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
        """Pushes and optionally concatenates a dataset to the Hugging Face Hub."""
        logger.info(f"Preparing to upload dataset to Hugging Face Hub at: {repo_id}")
        
        try:
            logger.info("Checking for existing dataset on the Hub...")
            existing_dataset = load_dataset(repo_id, split="train", token=hf_token)
            logger.info(f"Found existing dataset with {len(existing_dataset)} rows. Merging with new data...")
            final_dataset = concatenate_datasets([existing_dataset, new_dataset])
            
        except (RepositoryNotFoundError, ValueError, FileNotFoundError):
            logger.info("No existing dataset found (or repository is empty). Initializing as a new dataset.")
            final_dataset = new_dataset
            
        except Exception as e:
            logger.error(f"Failed to load existing dataset due to an unexpected error: {e}")
            raise

        logger.info(f"Pushing dataset with {len(final_dataset)} total rows to the Hub...")
        final_dataset.push_to_hub(repo_id, token=hf_token)
        logger.info("Successfully pushed dataset to the Hugging Face Hub!")


if __name__ == "__main__":
    # Set up argparse for CLI execution
    parser = argparse.ArgumentParser(description="Run the AksharAI Document Pipeline using a JSON config file.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the JSON configuration file (e.g., config.json)"
    )
    args = parser.parse_args()

    # Read the JSON config file
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        exit(1)
        
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON config file: {e}")
        exit(1)

    # Extract configuration variables (with safe defaults)
    PDF_FILE = config.get("pdf_path")
    OUTPUT_DIR = config.get("output_image_dir", "temp_images")
    DPI = config.get("dpi", 300)
    LANGUAGES = config.get("languages", ["guj", "eng"])
    MAX_WORKERS = config.get("max_ocr_workers", 4)
    HF_REPO_ID = config.get("hf_repo_id")

    # Retrieve HF Token securely from Environment Variables
    # If not found, it will default to None (which falls back to locally cached CLI credentials if available)
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        logger.warning("HF_TOKEN environment variable not found. Hugging Face upload may fail if you are not logged in via CLI.")

    # Validate essential config parameters
    if not PDF_FILE:
        logger.error("Configuration file MUST include a valid 'pdf_path'.")
        exit(1)

    # Initialize the pipeline
    pipeline = DocumentPipeline(languages=LANGUAGES, max_ocr_workers=MAX_WORKERS)
    
    try:
        # 1. Run the extraction pipeline
        extracted_document = pipeline.process_pdf(
            pdf_path=PDF_FILE, 
            output_image_dir=OUTPUT_DIR, 
            dpi=DPI
        )
        
        # 2. Convert to Hugging Face Dataset
        hf_dataset = pipeline.export_to_hf_dataset(
            document_results=extracted_document, 
            source_name=os.path.basename(PDF_FILE)
        )
        
        # 3. Concatenate and Push to Hub
        if len(hf_dataset) > 0:
            if HF_REPO_ID:
                pipeline.push_and_concatenate_to_hub(
                    new_dataset=hf_dataset, 
                    repo_id=HF_REPO_ID, 
                    hf_token=HF_TOKEN
                )
            else:
                logger.warning("No 'hf_repo_id' specified in config. Saving dataset locally instead.")
                hf_dataset.save_to_disk(f"{OUTPUT_DIR}_dataset")
        else:
            logger.warning("No text extracted. Aborting Hub upload.")
                
    except FileNotFoundError as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}")