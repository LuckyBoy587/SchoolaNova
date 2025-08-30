import pdfplumber
import nltk
import re

class PDFReader():
    def __init__(self, path):
        self.path = path

    def extract_text_by_cropping(self):
        """
        Extracts text from a two-column PDF by virtually cropping each half.

        Args:
            pdf_path (str): The file path to the PDF book.

        Returns:
            str: A single string with the text correctly ordered.
        """
        full_text = ""
        try:
            # Open the PDF file with pdfplumber
            with pdfplumber.open(self.path) as pdf:
                # Iterate through each page

                for page in pdf.pages:

                    # Get the dimensions of the page
                    width = page.width
                    height = page.height

                    # 1. Define the bounding box for the left column
                    # Bounding box is defined as (x0, top, x1, bottom)
                    left_bbox = (0, 0, width * 0.5, height) 
                    
                    # Crop the page to the left column's bounding box
                    left_column = page.crop(left_bbox)
                    
                    # Extract text from the cropped left column
                    left_text = left_column.extract_text()

                    # 2. Define the bounding box for the right column
                    right_bbox = (width * 0.5, 0, width, height)
                    
                    # Crop the page to the right column's bounding box
                    right_column = page.crop(right_bbox)
                    
                    # Extract text from the cropped right column
                    right_text = right_column.extract_text()

                    # 3. Combine the text in the correct order
                    if left_text:
                        full_text += left_text
                    if right_text:
                        full_text += right_text

        except Exception as e:
            print(f"An error occurred: {e}")
            return ""
            
        return full_text


    def remove_noises(self, book_content):

        # remove spaces
        cleaned_text = re.sub(r'\s+', ' ', book_content).strip()

        # A more advanced step could be to de-hyphenate words
        cleaned_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', cleaned_text)

        # 1. Protect the real paragraph breaks by replacing them with a unique placeholder
        text_with_placeholders = book_content.replace('\n\n', '__PARAGRAPH_BREAK__')

        # 2. Now, replace all remaining (unwanted) single line breaks with a space
        text_single_lines = text_with_placeholders.replace('\n', ' ')

        # 3. Finally, restore the real paragraph breaks
        cleaned_text = text_single_lines.replace('__PARAGRAPH_BREAK__', '\n\n')

        return cleaned_text

    def chunck_the_extracted_text(self, book_content):

        try:
            nltk.data.find('tokenizers/punkt')
        except:
            nltk.download('punkt')

        # split the entire string to meaningfull sentences
        sentences = nltk.sent_tokenize(book_content)

        # group the sentences into chunk for bettern understanding of models
        lines = []
        n=100
        cur_chunk = ""

        for sentence in sentences:
            cur_chunk += " " + sentence

            if len(cur_chunk.split()) >= n:
                lines.append(cur_chunk)
                cur_chunk = ""

        if cur_chunk:
            lines.append(cur_chunk)
        
        print(f"Total {len(lines)} Chunks extracted")
        return lines
    
    def get_content_from_pdf(self):

        book_content = self.extract_text_by_cropping(self.path)

        book_content = self.remove_noises(book_content=book_content)

        chunks = self.chunck_the_extracted_text(book_content=book_content)
