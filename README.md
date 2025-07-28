
# AI-Powered Financial Document Intelligence System

An end-to-end web application for extracting, classifying, and analyzing financial data from receipts and statements using advanced OCR and NLP techniques. This project provides a seamless workflow from document upload to insightful analytics, all in a modern, visually appealing dashboard.

---

## 🚀 Features

- **Receipt & Statement Upload:** Upload images (JPG, PNG) of receipts or PDF/CSV bank statements.
- **OCR Extraction:** Uses Tesseract OCR for accurate text extraction from images.
- **NLP Categorization:** Fine-tuned BERT model predicts expense categories from extracted text.
- **Interactive Dashboard:** Visualize spending by category, merchant, date, and more with advanced filters.
- **Advanced Filtering & Search:** Filter transactions by date, category, merchant, and source; search by keyword.
- **Modern UI:** Responsive, mobile-friendly design with a vibrant color scheme and smooth user experience.
- **Downloadable Reports:** Export recent transactions as CSV or Excel.
- **Accessibility:** Accessible forms and navigation for all users.

---

## 🏗️ File & Folder Structure

```
├── app/
│   ├── main.py              # Flask backend: API endpoints, routing, and app logic
│   ├── ocr_engine.py        # OCR logic using pytesseract
│   ├── nlp_model.py         # BERT-based classifier for expense categorization
│   ├── db_handler.py        # PostgreSQL database operations
│   ├── statement_parser.py  # PDF/CSV parsing and transaction extraction
│   └── templates/
│       ├── dashboard.html   # Dashboard UI (charts, filters, tables)
│       └── upload.html      # Upload UI (receipts/statements)
├── model/
│   └── fine_tuned_bert/     # Trained BERT model files (config, tokenizer, weights)
├── data/
│   ├── sample_receipts/     # Example receipt images for testing
│   └── sample_statements/   # Example statements (CSV)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

### File/Folder Descriptions

- **app/main.py**: Entry point for the Flask app; handles routing, API endpoints, and integration of all modules.
- **app/ocr_engine.py**: Contains functions for extracting text from images using Tesseract OCR.
- **app/nlp_model.py**: Loads and uses a fine-tuned BERT model to predict categories for transactions.
- **app/db_handler.py**: Manages database connections and CRUD operations for transactions.
- **app/statement_parser.py**: Parses PDF/CSV statements, extracts transactions, and applies OCR if needed.
- **app/templates/**: HTML templates for the dashboard and upload pages, styled with Bootstrap and custom CSS.
- **model/fine_tuned_bert/**: Directory for the trained BERT model, tokenizer, and config files.
- **data/sample_receipts/**: Example receipt images for testing OCR and upload features.
- **data/sample_statements/**: Example CSV statements for testing statement parsing.
- **requirements.txt**: Lists all Python dependencies required to run the project.
- **README.md**: This documentation file.

---

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AI-Powered-Financial-Document-Intelligence-System
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**
   ```bash
   python app/main.py
   ```
4. **Access the app**
   Open your browser at [http://localhost:5000](http://localhost:5000)

---

## 📊 Usage

1. **Upload a receipt or statement** via the Upload page.
2. **View extracted data and predicted categories** instantly after upload.
3. **Explore the Dashboard** for interactive charts, filters, and recent transactions.
4. **Export your data** as CSV/Excel for further analysis.

---

## 🛠️ Technologies Used

- Python 3, Flask
- Tesseract OCR (pytesseract)
- HuggingFace Transformers (BERT)
- Pandas, NumPy
- Chart.js, Bootstrap 5
- PostgreSQL (psycopg2)

---

## ✨ Customization & Extensibility

- Add more training data in `app/nlp_model.py` to improve category prediction.
- Adjust UI in `app/templates/` for branding or new features.
- Extend dashboard with new charts, analytics, or export options.
- Integrate user authentication for multi-user support.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT License

---

*Created by Purnendu Kale.*
