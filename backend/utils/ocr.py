import numpy as np

class OCRWorker:
    """
    Consolidated OCR utility using EasyOCR.
    """
    def __init__(self, reader):
        self.reader = reader

    def extract_text(self, img_np):
        """Standard OCR extraction."""
        if not self.reader: return []
        return self.reader.readtext(img_np)

    def find_axis_labels(self, img_np, axes_info):
        """Specific logic to find labels near axes."""
        # Logic to be implemented or called by orchestrator
        pass
