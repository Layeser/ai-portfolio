from pypdf import PdfReader


from backend.core.config import settings
from backend.utils.pdf_loader import load_pdf
def load_data():
    return load_pdf(settings.PDF_PATH)