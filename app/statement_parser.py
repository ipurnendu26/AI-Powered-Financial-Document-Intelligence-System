import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_csv(filepath: str) -> pd.DataFrame:
    """Parse CSV file and return DataFrame"""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully parsed CSV: {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error parsing CSV {filepath}: {str(e)}")
        return pd.DataFrame()


def parse_pdf(filepath: str) -> pd.DataFrame:
    """
    Try to parse tables from PDF using pdfplumber.
    Fallback to OCR (Tesseract) if no tables are found.
    """
    try:
        # Try pdfplumber first
        data = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table and len(table) > 1:
                    headers = table[0]
                    for row in table[1:]:
                        if len(row) == len(headers):
                            data.append(dict(zip(headers, row)))
        
        if data:
            df = pd.DataFrame(data)
            logger.info(f"Extracted {len(df)} rows from PDF using pdfplumber: {filepath}")
            return df
        else:
            logger.warning(f"pdfplumber found no tables. Trying OCR fallback on: {filepath}")
            return _parse_pdf_ocr_fallback(filepath)

    except Exception as e:
        logger.error(f"Failed to parse PDF {filepath}: {e}", exc_info=True)
        return pd.DataFrame()


def _parse_pdf_ocr_fallback(filepath: str) -> pd.DataFrame:
    """Fallback OCR parsing for PDF files"""
    try:
        # OCR fallback using pdf2image + pytesseract
        pages = convert_from_path(filepath, dpi=300)
        all_text = ""
        for i, page_img in enumerate(pages):
            text = pytesseract.image_to_string(page_img)
            logger.info(f"OCR text from page {i+1}:\n{text[:300]}...")  # Log preview
            all_text += text + "\n"

        # Very basic structuring: each line becomes a row
        lines = [line.strip() for line in all_text.split('\n') if line.strip()]
        columns = ['date', 'name', 'mode', 'amount', 'drcr']  # Assume structure
        rows = []
        for line in lines:
            parts = line.split()  # You can replace with regex
            if len(parts) >= 5:
                rows.append(parts[:5])  # Adjust based on your structure

        df = pd.DataFrame(rows, columns=columns)
        logger.info(f"OCR fallback extracted {len(df)} rows from scanned PDF")
        return df

    except Exception as e:
        logger.error(f"OCR fallback failed for PDF {filepath}: {e}", exc_info=True)
        return pd.DataFrame()


def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize financial data"""
    if df.empty:
        return df

    # Remove empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Try to infer columns if common structure isn't found
    if 'amount' not in df.columns and df.shape[1] >= 4:
        df.columns = ['date', 'name', 'mode', 'amount', 'drcr'][:len(df.columns)]

    # Handle NaNs
    df = df.replace({np.nan: None})

    # Convert amount and balance columns
    for col in ['amount', 'balance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert date columns to string (for JSON)
    for col in ['date']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df
