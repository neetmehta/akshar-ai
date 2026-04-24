"""
AksharAI Document Extraction Pipeline (CLI, Config & Directory Driven)

This module processes an entire folder of mixed documents (PDFs, JPGs, PNGs).
It reads settings from a JSON config file, securely loads the Hugging Face token 
from environment variables, and pushes the generated dataset to the Hub.
"""

import os
import cv2
import json
import logging
import argparse
from pathlib import Path
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
    A scalable pipeline to extract text from a directory of mixed document formats
    and sync them to Hugging Face Datasets.
    """
    
    SUPPORTED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}
    SUPPORTED_PDF_EXTS = {'.pdf'}

    def __init__(self, languages: List[str] = ['guj', 'eng'], max_ocr_workers: int = 4):
        logger.info("Initializing DocumentPipeline...")
        self.languages = languages
        self.max_ocr_workers = max_ocr_workers
        
        logger.info("Loading YOLOv10 Layout Model...")
        self.cropper = ParagraphCropper()
        logger.info("Pipeline initialized successfully.")

    def process_directory(self, input_dir: str, output_image_dir: str = "temp_images", dpi: int = 300) -> List[Dict[str, Any]]:
        """Scans a directory for supported files (PDFs and Images) and processes all of them."""
        input_path = Path(input_dir)
        if not input_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {input_dir}")

        logger.info(f"Scanning directory: {input_dir}")
        all_results = []

        for file_path in input_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.SUPPORTED_PDF_EXTS or ext in self.SUPPORTED_IMAGE_EXTS:
                    file_results = self.process_file(str(file_path), output_image_dir, dpi)
                    all_results.extend(file_results)

        logger.info(f"Finished processing directory. Extracted data from {len(all_results)} total pages/images.")
        return all_results

    def process_file(self, file_path: str, output_image_dir: str = "temp_images", dpi: int = 300) -> List[Dict[str, Any]]:
        """Routes a single file to the appropriate extraction logic based on its extension."""
        path_obj = Path(file_path)
        ext = path_obj.suffix.lower()
        file_name = path_obj.name
        
        logger.info(f"--- Starting pipeline for file: {file_name} ---")
        document_results = []

        try:
            if ext in self.SUPPORTED_PDF_EXTS:
                page_image_paths = split_pdf_to_images(file_path, output_dir=output_image_dir, dpi=dpi)
                
                for page_num, image_path in enumerate(page_image_paths):
                    logger.info(f"Processing {file_name} - Page {page_num + 1}/{len(page_image_paths)}...")
                    page_data = self._process_single_page(image_path, page_num + 1, source_name=file_name)
                    document_results.append(page_data)
                    
            elif ext in self.SUPPORTED_IMAGE_EXTS:
                logger.info(f"Processing image file {file_name} directly...")
                page_data = self._process_single_page(file_path, page_num=1, source_name=file_name)
                document_results.append(page_data)

            return document_results

        except Exception as e:
            logger.error(f"Pipeline failed during execution of {file_name}: {e}")
            return document_results

    def _process_single_page(self, image_path: str, page_num: int, source_name: str) -> Dict[str, Any]:
        """Processes a single image page for paragraphs and OCR."""
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to read image at {image_path}. Skipping page.")
            return {"source": source_name, "page": page_num, "paragraphs": []}

        crops = self.cropper.crop_paragraphs(image, threshold=0.85)
        logger.info(f"{source_name} [Page {page_num}]: Detected {len(crops)} paragraphs.")

        extracted_texts = self._parallel_ocr(crops)

        return {
            "source": source_name,
            "page": page_num,
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

    def export_to_hf_dataset(self, document_results: List[Dict[str, Any]]) -> Dataset:
        """Converts extracted text across multiple documents into a Hugging Face Dataset."""
        logger.info("Converting extracted data to Hugging Face Dataset format...")
        
        hf_data_dict = {
            "source_document": [],
            "page_number": [],
            "paragraph_index": [],
            "text": []
        }
        
        for page_data in document_results:
            source_name = page_data["source"]
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
    INPUT_DIR = config.get("input_directory")
    OUTPUT_DIR = config.get("output_image_dir", "temp_images")
    DPI = config.get("dpi", 300)
    LANGUAGES = config.get("languages", ["guj", "eng"])
    MAX_WORKERS = config.get("max_ocr_workers", 4)
    HF_REPO_ID = config.get("hf_repo_id")

    # Retrieve HF Token securely from Environment Variables
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        logger.warning("HF_TOKEN environment variable not found. Hugging Face upload may fail if you are not logged in via CLI.")

    # Validate essential config parameters
    if not INPUT_DIR:
        logger.error("Configuration file MUST include a valid 'input_directory'.")
        exit(1)

    # Initialize the pipeline
    pipeline = DocumentPipeline(languages=LANGUAGES, max_ocr_workers=MAX_WORKERS)
    
    try:
        # 1. Run the extraction pipeline on the entire directory
        extracted_data = pipeline.process_directory(
            input_dir=INPUT_DIR, 
            output_image_dir=OUTPUT_DIR, 
            dpi=DPI
        )
        
        # 2. Convert to Hugging Face Dataset
        hf_dataset = pipeline.export_to_hf_dataset(
            document_results=extracted_data
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
                
    except NotADirectoryError as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}")