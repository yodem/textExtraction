# Text Extractor

This project extracts Hebrew text from PDF files, processes it, and saves the extracted text to files. It uses Tesseract OCR for text recognition and various image processing techniques to detect and extract text blocks.

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare the environment:**

   - Ensure Tesseract OCR is installed and configured correctly.
   - Install the required Python packages using the provided `requirements.txt`.

2. **Run the script:**

   - Place your PDF files in the `pdf_files` directory.
   - Run the main script to process the PDF files and extract text:
     ```bash
     python multiple_cols.py
     ```

3. **Output:**
   - The extracted text will be saved in the `extractedText` directory with the same name as the PDF file but with a `.txt` extension.

## Functions

### `visualize_columns(pdf_path, output_dir='column_visualization4')`

Draws rectangles around detected columns using image processing techniques and saves the visualizations.

### `clean_hebrew_text(text)`

Cleans Hebrew text by removing non-Hebrew characters and normalizing spaces.

### `process_text_blocks(pdf_path, blocks, page, cv_img)`

Processes text blocks from a page and extracts text in the correct order.

### `extract_rashi_text(pdf_path)`

Extracts Hebrew text from a PDF file by processing text blocks if found, otherwise extracts the full page.

### `save_to_file(text_list, output_path)`

Saves extracted text to a file with proper RTL encoding.

### `process_directory(input_dir='.', output_dir='extracted_texts')`

Processes all PDF files in the given directory and saves the extracted texts.

### `main()`

Main function to run the script.

## Example

To process PDF files and extract text, follow these steps:

1. Place your PDF files in the `pdf_files` directory.
2. Run the script:
   ```bash
   python multiple_cols.py
   ```
3. Check the `extractedText` directory for the extracted text files.

## License

This project is licensed under the MIT License.
