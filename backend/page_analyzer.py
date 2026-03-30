import fitz
import json
import os
import re
import argparse
import base64
import cv2
import numpy as np
import easyocr
from llm_validator import evaluate_graph_image_llm
from schema_loader import load_domain_schema

class PageAnalyzer:
    """
    Analyzes PDF pages to detect tables, figures, and dense numerical text.
    V2 Architecture: Generates continuous confidence scores, filters non-scientific sections,
    and runs aggressive local heuristics before invoking LLM Vision APIs to preserve tokens.
    """
    def __init__(self, workspace=None):
        self.schema = load_domain_schema(workspace=workspace)
        self.domain_units = self._extract_domain_units()
        self.domain_attributes = [k.lower() for k in self.schema.schema.keys()] if self.schema else []
        self.domain_aliases = []
        if self.schema:
            for v in self.schema.schema.values():
                self.domain_aliases.extend([a.lower() for a in v.get("aliases", [])])
        # Lazy load OCR to save VRAM if never used
        self.reader = None

    def _extract_domain_units(self):
        units = set()
        if self.schema and hasattr(self.schema, 'schema'):
             for v in self.schema.schema.values():
                 units.update([u.lower() for u in v.get("units", [])])
        return units

    def analyze_pdf(self, pdf_path):
        """
        Analyzes all pages in a PDF and returns a list of page routing metadata.
        """
        results = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text().lower()
                    
                    # 1. Section Filter (Improvement 4)
                    # Forcibly reject references, bibliographies, and acknowledgements
                    if self._is_ignored_section(text, page_num, len(doc)):
                        results.append({
                            "page": page_num,
                            "table_score": 0.0,
                            "plot_score": 0.0,
                            "text_score": 0.0,
                            "table_count": 0,
                            "image_count": 0,
                            "reason": "Ignored Section (References/Ack)"
                        })
                        continue

                    # 2. Detect Tables (V2 Score)
                    tables = page.find_tables()
                    table_count = len(tables.tables) if tables else 0
                    table_score = 0.0
                    
                    if table_count > 0:
                        total_cells = 0
                        numeric_cells = 0
                        total_rows = 0
                        for t in tables.tables:
                            try:
                                extracted = t.extract()
                                if not extracted: continue
                                total_rows += len(extracted)
                                for row in extracted:
                                    total_cells += len(row)
                                    for cell in row:
                                        if cell and re.search(r'\d', str(cell)):
                                            numeric_cells += 1
                            except Exception:
                                pass
                        
                        if total_cells > 0 and total_rows > 0:
                            num_ratio = numeric_cells / total_cells
                            avg_row_len = total_cells / total_rows
                            # A real scientific table usually has >3 columns, high numeric ratio
                            width_bonus = min(0.2, (avg_row_len - 2) * 0.05) if avg_row_len > 2 else 0.0
                            table_score = min(1.0, 0.3 + (table_count * 0.1) + (num_ratio * 0.5) + width_bonus)

                    # 3. Detect Figures & Images (Improvements 1 & 2 & 3.2)
                    images = page.get_images()
                    plot_score = 0.0
                    
                    # Get drawn vectors (charts without raster backgrounds)
                    try:
                        drawings = page.get_drawings()
                        long_lines = 0
                        for d in drawings:
                            for item in d.get("items", []):
                                if item[0] == "l": # line
                                    p1, p2 = item[1], item[2]
                                    if abs(p1.x - p2.x) < 2 and abs(p1.y - p2.y) > 50:
                                        long_lines += 1
                                    elif abs(p1.y - p2.y) < 2 and abs(p1.x - p2.x) > 50:
                                        long_lines += 1
                        if long_lines >= 2 and len(drawings) > 20: 
                             plot_score = max(plot_score, 0.4) # Base confidence for vector chart with axes
                    except Exception:
                        pass
                        
                    for img in images:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        # Filter 1: Size
                        if pix.width < 150 or pix.height < 150:
                            pix = None
                            continue
                            
                        # Convert to CV2 numpy array natively from PyMuPDF uncompressed memory
                        # (Bypasses PyMuPDF compression panics: ValueError: '{output}' cannot have alpha)
                        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                        
                        if pix.n == 4: # RGBA -> BGR
                            cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                        elif pix.n == 3: # RGB -> BGR
                            cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        elif pix.n == 1: # GRAY -> BGR
                            cv_img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                        else:
                             # Failsafe: convert alien color spaces to RGB then drop to OpenCV
                             temp_pix = fitz.Pixmap(fitz.csRGB, pix)
                             img_np = np.frombuffer(temp_pix.samples, dtype=np.uint8).reshape(temp_pix.h, temp_pix.w, temp_pix.n)
                             cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                             temp_pix = None
                        
                        # Set img_data for base64 Vision LLM Oracle if heuristics pass
                        img_data = cv2.imencode('.jpg', cv_img)[1].tobytes()
                        
                        # Filter 2: Local Line/Axis Heuristics
                        has_lines = self._has_axis_lines(cv_img)
                        
                        # Filter 3: Local OCR Heuristics
                        has_numbers = False
                        if has_lines:
                             has_numbers = self._has_ocr_numbers(cv_img)
                             
                        # Filter 4: LLM Vision Oracle
                        if has_lines and has_numbers:
                            b64 = base64.b64encode(img_data).decode('utf-8')
                            print(f"    [PageAnalyzer] Image ({pix.width}x{pix.height}) passed local heuristics. Calling LLM...")
                            
                            try:
                                vision_res = evaluate_graph_image_llm(b64)
                                is_graph = vision_res.get("is_graph", False)
                                llm_conf = vision_res.get("confidence", 0.0)
                            except Exception as e:
                                print(f"    [PageAnalyzer] LLM Vision failed ({e}). Defaulting to heuristic.")
                                is_graph = False
                                llm_conf = 0.0
                            
                            local_feat_score = (1.0 if has_lines else 0.0) * 0.5 + (1.0 if has_numbers else 0.0) * 0.5
                            
                            if is_graph:
                                final_img_score = (0.6 * llm_conf) + (0.4 * local_feat_score)
                            else:
                                final_img_score = local_feat_score * 0.4 # Heavily discount if LLM rejected or failed
                                
                            plot_score = max(plot_score, final_img_score)
                        
                        pix = None # Free memory
                    
                    # 4. Dense Numerical Text (Improvement 3.4)
                    text_score = self._calculate_scientific_density(text)
                    
                    results.append({
                        "page": page_num,
                        "table_score": table_score,
                        "plot_score": plot_score,
                        "text_score": text_score,
                        "table_count": table_count,
                        "image_count": len(images)
                    })
        except Exception as e:
            print(f"Error analyzing PDF {pdf_path}: {e}")
            
        return results

    def _is_ignored_section(self, text, page_num, total_pages):
        """Ignore trailing reference pages to prevent hallucination extraction."""
        # Only strict about the last 20% of the paper
        if page_num > total_pages * 0.8:
            if "references" in text[-1000:] or "bibliography" in text[-1000:]:
                return True
        if "acknowledgement" in text:
            # Often near the end, signals the end of the data section
            return True
        return False

    def _has_axis_lines(self, cv_img):
        """Uses fast Canny edge detection and probabilistic Hough lines to determine if the image looks like a chart."""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Look for somewhat long straight lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
        return lines is not None and len(lines) >= 2

    def _has_ocr_numbers(self, cv_img):
        """Checks if the image has aligned numeric axis labels before sending to Vision API."""
        if self.reader is None:
            self.reader = easyocr.Reader(['en'], gpu=True) # Will fallback to CPU automatically
            
        results = self.reader.readtext(cv_img, detail=1)
        
        centers_x = []
        centers_y = []
        for bbox, text, conf in results:
            if re.match(r'^-?\d+(?:\.\d+)?(?:e[-+]?\d+)?$', text.strip(), re.IGNORECASE):
                cx = (bbox[0][0] + bbox[1][0]) / 2
                cy = (bbox[0][1] + bbox[2][1]) / 2
                centers_x.append(cx)
                centers_y.append(cy)
                
        if len(centers_x) < 3:
            return False
            
        def get_max_aligned(coords, tolerance=15):
            if not coords: return 0
            max_aligned = 0
            for c1 in coords:
                aligned = sum(1 for c2 in coords if abs(c1 - c2) < tolerance)
                if aligned > max_aligned:
                    max_aligned = aligned
            return max_aligned
            
        # A valid graph must have at least 3 aligned numbers on an axis (ticks)
        return get_max_aligned(centers_x, 15) >= 3 or get_max_aligned(centers_y, 15) >= 3

    def _calculate_scientific_density(self, text_lower):
        """Searches for pattern matching <number> + <valid scientific unit> and proximity to keywords."""
        if not text_lower or not self.domain_units:
             return 0.0
             
        score = 0.0
        unit_pattern = "|".join([re.escape(u) for u in self.domain_units])
        if not unit_pattern: return 0.0
        
        # Base Match: Support numbers, scientific notation, ± values, and ranges
        number_component = r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
        sci_pattern = rf"\b{number_component}\s*(?:(?:-|–|to|±)\s*{number_component}\s*)?(?:{unit_pattern})\b"
        matches = list(re.finditer(sci_pattern, text_lower, re.IGNORECASE))
        
        if not matches:
             return 0.0
             
        # Context Match: Are these numbers near actual domain keywords?
        context_score = 0.0
        keyword_pattern = "|".join([re.escape(a) for a in self.domain_attributes + self.domain_aliases])
        
        for match in matches:
             # Grab 80 characters before and after the matched measurement
             start_idx = max(0, match.start() - 80)
             end_idx = min(len(text_lower), match.end() + 80)
             window = text_lower[start_idx:end_idx]
             
             if re.search(keyword_pattern, window):
                  context_score += 1.0 # High value if near keyword
             else:
                  context_score += 0.2 # Low value if isolated
        
        # Normalize score
        if context_score >= 8.0:
             score = 1.0
        elif context_score >= 4.0:
             score = 0.75
        elif context_score >= 1.0:
             score = 0.40
             
        return score

def main():
    parser = argparse.ArgumentParser(description="Analyze PDF pages for tables and plots.")
    parser.add_argument("--workspace", required=True, help="Workspace directory")
    parser.add_argument("--attributes", help="Comma-separated attributes (optional for analyzer)")
    args = parser.parse_args()

    workspace = args.workspace
    # Standardize paths
    project_root = os.path.dirname(os.path.dirname(workspace))
    task_id = os.path.basename(workspace)
    included_dir = os.path.join(project_root, "Included", task_id)
    output_json = os.path.join(workspace, "page_analysis_results.json")

    if not os.path.exists(included_dir):
        print(f"Included directory not found: {included_dir}")
        return

    analyzer = PageAnalyzer(workspace=workspace)
    analysis_results = []

    for filename in os.listdir(included_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(included_dir, filename)
            print(f"Analyzing {filename}...")
            pages_meta = analyzer.analyze_pdf(pdf_path)
            
            tables_pages = []
            plots_pages = []
            text_pages = []
            
            # Independent Threshold Routing (Multi-Extract Support)
            for m in pages_meta:
                 page = m["page"]
                 t_score = m.get("table_score", 0.0)
                 p_score = m.get("plot_score", 0.0)
                 txt_score = m.get("text_score", 0.0)
                 
                 # Table Routing
                 if t_score >= 0.4:
                      tables_pages.append(page)
                 
                 # Plot Routing
                 if p_score >= 0.4:
                      plots_pages.append(page)
                      
                 # Text Routing (Lowered threshold from 0.4 to 0.25)
                 if txt_score >= 0.25:
                      text_pages.append(page)
            
            analysis_results.append({
                "file": filename,
                "tables_pages": sorted(list(set(tables_pages))),
                "plots_pages": sorted(list(set(plots_pages))),
                "text_pages": sorted(list(set(text_pages)))
            })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Page analysis complete. Saved to {output_json}")

if __name__ == "__main__":
    main()
