import pdfplumber
import re
import os
from pathlib import Path
from pdf2image import convert_from_path
import cv2
import numpy as np


def visualize_columns(pdf_path, output_dir='column_visualization4'):
    """
    Draw rectangles around the detected columns using image processing
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        
        for i, img in enumerate(images):
            # Convert PIL image to OpenCV format
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get black and white image
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Create kernels for vertical and horizontal connections
            vertical_kernel = np.ones((40,1), np.uint8)
            horizontal_kernel = np.ones((1,20), np.uint8)
            
            # Connect text vertically first
            binary_v = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vertical_kernel)
            
            # Then connect horizontally
            binary = cv2.morphologyEx(binary_v, cv2.MORPH_CLOSE, horizontal_kernel)
            
            # Clean up noise
            kernel = np.ones((5,5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Dilate to merge nearby text blocks
            binary = cv2.dilate(binary, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and width
            min_area = cv_img.shape[0] * cv_img.shape[1] * 0.05  # Minimum 5% of page area
            text_blocks = []
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = h/w if w > 0 else 0
                
                # Filter blocks by size and aspect ratio
                if (area > min_area and 
                    w > cv_img.shape[1] * 0.2 and  # Width at least 20% of page width
                    aspect_ratio > 1.0):           # Height greater than width (column-like)
                    text_blocks.append((x, y, x+w, y+h))
            
            # Sort blocks by x-coordinate (right to left for Hebrew)
            text_blocks.sort(key=lambda x: x[0], reverse=True)
            
            # Draw the detected blocks
            result_img = cv_img.copy()
            for j, block in enumerate(text_blocks):
                color = (0, 0, 255) if j == 0 else (255, 0, 0)  # Red for right column, Blue for left
                cv2.rectangle(result_img, (block[0], block[1]), (block[2], block[3]), color, 2)
                
                # Add block number and dimensions
                text = f"Block {j+1}: {block[2]-block[0]}x{block[3]-block[1]}"
                cv2.putText(result_img, text, (block[0], block[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save debug images
            cv2.imwrite(os.path.join(output_dir, f'page_{i+1}_binary.png'), binary)
            cv2.imwrite(os.path.join(output_dir, f'page_{i+1}_vertical.png'), binary_v)
            cv2.imwrite(os.path.join(output_dir, f'page_{i+1}_blocks.png'), result_img)
            
            print(f"Saved visualization for page {i+1}")
            print(f"Found {len(text_blocks)} text blocks")
                
    except Exception as e:
        print(f"Error visualizing columns: {str(e)}")
        raise e

def clean_hebrew_text(text):
    """Clean Hebrew text before processing"""
    # Remove any non-Hebrew characters except common punctuation
    text = re.sub(r'[^\u0590-\u05FF\.,!?"\s]', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove any RTL/LTR marks
    text = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)
    return text.strip()

def process_text_blocks(pdf_path, blocks, page, cv_img):
    """
    Process text blocks from a page and extract text in correct order
    """
    page_text = []
    width = page.width
    height = page.height
    scale_x = width / cv_img.shape[1]
    scale_y = height / cv_img.shape[0]
    
    for i, block in enumerate(blocks):
        pdf_bbox = (
            block[0] * scale_x,
            block[1] * scale_y,
            block[2] * scale_x,
            block[3] * scale_y
        )
        
        try:
            crop = page.crop(bbox=pdf_bbox)
            if crop:
                text = crop.extract_text()
                if text:
                    text = clean_rashi_text(text)
                    # text = clean_hebrew_text(text)
                    print(text)
                    column_type = "Right Column" if i == 0 else "Left Column"
                    page_text.append(f"{'='*20}\n{column_type}\n{'='*20}")
                    page_text.append(text)
        except Exception as e:
            print(f"Error processing block {i+1}: {str(e)}")
    
    return page_text

def extract_rashi_text(pdf_path):
    """
    Extract Hebrew text from a PDF file - process text blocks if found, otherwise extract full page
    """
    extracted_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            
            for page_num, (page, img) in enumerate(zip(pdf.pages, images), 1):
                # Add page header
                extracted_text.append(f"\n{'#'*50}")
                extracted_text.append(f"Page {page_num}")
                extracted_text.append(f"{'#'*50}\n")
                
                # Try to detect and process columns
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold to get black and white image
                _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                
                # Create kernels for vertical and horizontal connections
                vertical_kernel = np.ones((40,1), np.uint8)
                horizontal_kernel = np.ones((1,20), np.uint8)
                
                # Connect text vertically first
                binary_v = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vertical_kernel)
                binary = cv2.morphologyEx(binary_v, cv2.MORPH_CLOSE, horizontal_kernel)
                
                # Clean up and merge text blocks
                kernel = np.ones((5,5), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                binary = cv2.dilate(binary, kernel, iterations=2)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter and sort text blocks
                min_area = cv_img.shape[0] * cv_img.shape[1] * 0.05
                text_blocks = []
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = h/w if w > 0 else 0
                    
                    if (area > min_area and 
                        w > cv_img.shape[1] * 0.2 and
                        aspect_ratio > 1.0):
                        text_blocks.append((x, y, x+w, y+h))
                
                # Process text blocks if any found, otherwise extract full page
                if text_blocks:
                    # Sort blocks right to left
                    text_blocks.sort(key=lambda x: x[0], reverse=True)
                    page_text = process_text_blocks(pdf_path, text_blocks, page, cv_img)
                    if page_text:  # If we got text from the blocks
                        extracted_text.extend(page_text)
                    else:  # If no text was extracted from blocks, fall back to full page
                        text = page.extract_text()
                        if text:
                            text = clean_rashi_text(text)
                            print(text)
                            extracted_text.append(f"{'='*20}\nFull Page Text\n{'='*20}")
                            extracted_text.append(text)
                else:
                    # Extract entire page if no blocks found
                    print(f"No text blocks found on page {page_num}, extracting full page")
                    text = page.extract_text()
                    if text:
                        text = clean_rashi_text(text)
                        print(text)
                        extracted_text.append(f"{'='*20}\nFull Page Text\n{'='*20}")
                        extracted_text.append(text)
                
                # Add page separator
                extracted_text.append(f"\n{'='*50}\n")
                
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return []
        
    return extracted_text

def save_to_file(text_list, output_path):
    """
    Save extracted text to a file with proper RTL encoding
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\u200F')  # RTL mark at start
            for text in text_list:
                if text.startswith(('#', '=')) or text.startswith('Page'):
                    # Don't reverse separators and page numbers
                    f.write(text + '\n')
                else:
                    # Split text into lines, reverse each line separately
                    lines = text.split('\n')
                    for line in lines:
                        if line.strip():  # Only process non-empty lines
                            f.write(line.strip()[::-1] + '\n')
                    f.write('\n')  # Add extra newline between blocks
    except Exception as e:
        print(f"Error saving file: {str(e)}")

def clean_rashi_text(text):
    """
    Additional cleaning specific to Rashi text
    """
    # Remove any Latin characters
    text = re.sub(r'[a-zA-Z]', '', text)
    
    # Remove any digits
    text = re.sub(r'\d', '', text)
    
    # Remove any special characters except Hebrew punctuation and selected punctuation marks
    text = re.sub(r'[^\u0590-\u05FF\u200f\u202a-\u202e\.,!?"\s]', '', text)
    
    # Remove any remaining RTL or LTR marks that might interfere
    text = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)
    
    return text

def process_directory(input_dir='.', output_dir='extracted_texts'):
    """
    Process all PDF files in the given directory
    
    Args:
        input_dir (str): Directory containing PDF files
        output_dir (str): Directory where extracted texts will be saved
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the directory")
        return
    
    for pdf_file in pdf_files:
        input_path = os.path.join(input_dir, pdf_file)
        # Create output filename by replacing .pdf with .txt
        output_filename = os.path.splitext(pdf_file)[0] + '.txt'
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing: {pdf_file}")
        
        # Extract text
        extracted_text = extract_rashi_text(input_path)
        
        if extracted_text:
            # Save to file
            save_to_file(extracted_text, output_path)
            print(f"Successfully extracted text to: {output_path}")
        else:
            print(f"No text was extracted from: {pdf_file}")

def main():
    input_dir = "./pdf_files"
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file}")
        pdf_path = os.path.join(input_dir, pdf_file)
        
        # Visualize columns first
        visualize_columns(pdf_path)
        
        output_dir = "./extractedText"
        process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()
