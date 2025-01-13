import os
import requests
from pathlib import Path
import logging
import re
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TextFixer:
    def __init__(self, input_dir="./extractedText", output_dir="./fixed_texts"):
        self.input_dir = input_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.server_url = "http://localhost:8000/fix-hebrew"
        self.max_chunk_size = 1000  # Increased chunk size
        self.request_delay = 2.0  # Increased delay between requests
        self.ignore_patterns = ['Right Column', 'Left Column', 'Full Page Text', 'Page', '==', '##']
        self.current_page = None
        self.page_content = []

    def fix_paragraph(self, text: str) -> str:
        """Send paragraph to server for correction with retry logic"""
        if not text.strip():
            return text
            
        for attempt in range(3):
            try:
                # Add delay between requests
                time.sleep(self.request_delay)
                
                response = requests.post(
                    self.server_url,
                    json={"text": text}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    fixed_text = result["fixed_text"]
                    print(f"Fixed text: {fixed_text}")
                    return fixed_text
                elif response.status_code == 429:  # Resource exhausted
                    logger.warning(f"Rate limit hit, attempt {attempt + 1}/3")
                    time.sleep(5 * (attempt + 1))  # Increased backoff
                    continue
                else:
                    logger.error(f"Server error: {response.text}")
                    return text
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt == 2:  # Last attempt
                    return text
                time.sleep(5)
        
        return text

    def clean_text_content(self, text: str) -> str:
        """Remove headers and keep only Hebrew text content"""
        lines = []
        for line in text.split('\n'):
            # Skip lines that match any of the ignore patterns
            if any(pattern in line for pattern in self.ignore_patterns):
                continue
            # Keep only lines with Hebrew characters
            if any('\u0590' <= c <= '\u05FF' for c in line):
                lines.append(line.strip())
        return '\n'.join(lines)

    def process_file(self, file_path: Path):
        """Process a single text file with page separation"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            current_page = None
            current_block = []
            self.page_content = []  # List to store content by page

            # Process content line by line
            for line in content.split('\n'):
                if line.startswith('Page '):
                    # Save previous page content if exists
                    if current_page and current_block:
                        self._process_page_block(current_block)
                        current_block = []
                    
                    try:
                        current_page = int(line.split()[1])
                        self.page_content.append({
                            'page_num': current_page,
                            'content': []
                        })
                    except ValueError:
                        pass
                    continue

                # Skip non-Hebrew content and headers
                if not any('\u0590' <= c <= '\u05FF' for c in line):
                    continue

                if line.strip():
                    current_block.append(line.strip())
                elif current_block:  # Empty line and we have content
                    self._process_page_block(current_block)
                    current_block = []

            # Process any remaining text
            if current_block:
                self._process_page_block(current_block)

            # Save fixed text with pages and 5 words per line
            output_path = self.output_dir / f"fixed_{file_path.name}"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\u200F')  # RTL mark
                
                for page in self.page_content:
                    # Write page header
                    f.write(f"\n{'#'*50}\n")
                    f.write(f"Page {page['page_num']}\n")
                    f.write(f"{'#'*50}\n\n")
                    
                    # Write page content
                    for paragraph in page['content']:
                        # Split into 5-word lines
                        words = paragraph.split()
                        for i in range(0, len(words), 5):
                            line_words = words[i:i+5]
                            f.write(' '.join(line_words) + '\n')
                        f.write('\n')  # Extra newline between paragraphs

            logger.info(f"Processed file saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    def _process_page_block(self, block):
        """Process a block of text and add it to current page"""
        if not block or not self.page_content:
            return
            
        text = '\n'.join(block)
        text = self.clean_text_content(text)
        if text:
            fixed_text = self.fix_paragraph(text)
            if fixed_text:
                self.page_content[-1]['content'].append(fixed_text)

    def process_directory(self):
        """Process all text files in the input directory"""
        input_path = Path(self.input_dir)
        files = list(input_path.glob('*.txt'))

        if not files:
            logger.warning(f"No text files found in {self.input_dir}")
            return

        for file in files:
            logger.info(f"Processing {file}")
            self.process_file(file)

def main():
    fixer = TextFixer(input_dir="./extractedText", output_dir="./fixedTexts")
    fixer.process_directory()

if __name__ == "__main__": 
    main()
