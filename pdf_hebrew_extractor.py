import pdfplumber
import re
import os
from pathlib import Path
import cv2
import numpy as np
from pdf2image import convert_from_path

def detect_text_blocks(pdf_path, page_num=0):
    """
    Detect text blocks using image processing
    """
    # Convert PDF page to image
    images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
    if not images:
        return None
    
    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get black and white image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Noise removal
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours by x-coordinate
    min_area = img.shape[0] * img.shape[1] * 0.05  # Minimum 5% of page area
    text_blocks = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            text_blocks.append((x, y, x+w, y+h))
    
    # Sort blocks by x-coordinate
    text_blocks.sort(key=lambda x: x[0])
    
    # Debug: Save visualization
    debug_img = img.copy()
    for block in text_blocks:
        cv2.rectangle(debug_img, (block[0], block[1]), (block[2], block[3]), (0, 255, 0), 2)
    cv2.imwrite(f'debug_blocks_page_{page_num}.png', debug_img)
    
    return text_blocks

def find_column_boundaries(page, pdf_path):
    """
    Detect column boundaries using both image processing and text analysis
    """
    # Get page dimensions
    width = page.width
    height = page.height
    
    # Detect text blocks using image processing
    text_blocks = detect_text_blocks(pdf_path, page.page_number - 1)
    
    if not text_blocks or len(text_blocks) < 2:
        print(f"Could not detect text blocks on page {page.page_number}")
        return None
    
    # Calculate scaling factor between image and PDF coordinates
    img = convert_from_path(pdf_path, first_page=page.page_number, last_page=page.page_number)[0]
    scale_x = width / img.width
    scale_y = height / img.height
    
    # Convert image coordinates to PDF coordinates
    pdf_blocks = []
    for block in text_blocks:
        pdf_block = (
            block[0] * scale_x,
            block[1] * scale_y,
            block[2] * scale_x,
            block[3] * scale_y
        )
        pdf_blocks.append(pdf_block)
    
    # Assuming two main text blocks, use their boundaries
    if len(pdf_blocks) >= 2:
        right_block = pdf_blocks[0]  # First block (rightmost in RTL)
        left_block = pdf_blocks[1]   # Second block
        
        margin = 5
        
        right_bbox = (
            right_block[0] - margin,
            margin,
            right_block[2] + margin,
            height - margin
        )
        
        left_bbox = (
            left_block[0] - margin,
            margin,
            left_block[2] + margin,
            height - margin
        )
        
        return left_bbox, right_bbox
    
    return None

def extract_rashi_text(pdf_path):
    """
    Extract Hebrew text from a two-column PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of extracted text segments
    """
    extracted_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Detect column boundaries using image processing
                boundaries = find_column_boundaries(page, pdf_path)
                
                if not boundaries:
                    print(f"Could not detect columns on page {page.page_number}")
                    continue
                    
                left_bbox, right_bbox = boundaries
                
                try:
                    # Process right column (first half)
                    right_crop = page.crop(bbox=right_bbox)
                    if right_crop:
                        right_text = right_crop.extract_text()
                        if right_text:
                            right_text = clean_rashi_text(right_text)
                            extracted_text.append("="*20 + "\nRight Column\n" + "="*20)
                            extracted_text.append(right_text)
                except Exception as column_error:
                    print(f"Error processing right column: {str(column_error)}")
                
                try:
                    # Process left column (second half)
                    left_crop = page.crop(bbox=left_bbox)
                    if left_crop:
                        left_text = left_crop.extract_text()
                        if left_text:
                            left_text = clean_rashi_text(left_text)
                            extracted_text.append("="*20 + "\nLeft Column\n" + "="*20)
                            extracted_text.append(left_text)
                except Exception as column_error:
                    print(f"Error processing left column: {str(column_error)}")
                
                # Add page separator
                extracted_text.append("-"*40 + f"\nPage {page.page_number}\n" + "-"*40)
                    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return []
        
    return extracted_text

def save_to_file(text_list, output_path):
    """
    Save extracted text to a file with proper RTL encoding
    
    Args:
        text_list (list): List of text segments to save
        output_path (str): Path where to save the output file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Add RTL mark at the start of the file
            f.write('\u200F')
            for text in text_list:
                # Add RTL mark before each line and reverse the text
                f.write('\u200F' + text[::-1] + '\n\n')
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
    
    # Remove any special characters except Hebrew punctuation
    text = re.sub(r'[^\u0590-\u05FF\u200f\u202a-\u202e\.,\s]', '', text)
    
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
    # Example with custom directories
    input_dir = "./pdf_files"  # Your PDFs directory
    output_dir = "./hebrew_texts"  # Where to save extracted texts
    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main() 