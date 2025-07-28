import os
import cv2
import logging
import pytesseract

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """
    Apply preprocessing like grayscale + thresholding (if needed).
    Saves the preprocessed image with a standard name in same folder.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Generate consistent preprocessed image path
        folder = os.path.dirname(image_path)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        preprocessed_path = os.path.join(folder, f"{filename}_preprocessed.jpg")

        cv2.imwrite(preprocessed_path, thresh)
        return preprocessed_path

    except Exception as e:
        logger.error(f"Image preprocessing failed for {image_path}: {str(e)}")
        return None


def extract_text(image_path):
    """
    Main OCR pipeline that extracts text from an image using Tesseract.
    """
    try:
        logger.info(f"🔍 Starting OCR for: {image_path}")

        preprocessed_path = preprocess_image(image_path)
        if not preprocessed_path or not os.path.exists(preprocessed_path):
            raise FileNotFoundError(f"Preprocessed image not found: {preprocessed_path}")

        # Use pytesseract to extract text
        full_text = pytesseract.image_to_string(preprocessed_path, lang='eng')
        full_text = full_text.strip()

        logger.info(f"✅ OCR extracted text: {full_text[:100]}...")  # Preview first 100 chars
        return full_text

    except Exception as e:
        logger.error(f"OCR extraction failed for {image_path}: {str(e)}")
        return ""
