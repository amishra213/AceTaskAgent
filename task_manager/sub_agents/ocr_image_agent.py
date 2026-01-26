"""
OCR and Image Extractor Sub-Agent for handling image-based data extraction.

Capabilities:
- Extract text from images using OCR (Tesseract, EasyOCR, PaddleOCR)
- Extract images from documents (PDFs, Word, etc.)
- Process screenshots and scanned documents
- Recognize text in multiple languages
- Extract structured data from forms and tables in images

Migration Status: Week 7 Day 1 - Dual Format Support
- Renamed run_operation → execute_task
- Supports both legacy dict and standardized AgentExecutionRequest/Response
- Maintains 100% backward compatibility
- Publishes SystemEvent on completion for event-driven workflows
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import json
import base64
import numpy as np
import time

from task_manager.utils.logger import get_logger

# Import standardized schemas and utilities (Week 1-2 implementation)
from task_manager.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    create_system_event,
    create_error_response
)
from task_manager.utils import (
    auto_convert_response,
    validate_agent_execution_response,
    exception_to_error_response,
    InvalidParameterError,
    OCRError,
    MissingDependencyError,
    wrap_exception
)
from task_manager.core.event_bus import get_event_bus

logger = get_logger(__name__)


class OCRImageAgent:
    """
    Sub-agent for OCR and image extraction operations.
    
    This agent handles all image and OCR-related tasks:
    - Extracting text from images using OCR
    - Extracting images from documents (PDF, Word)
    - Processing scanned documents
    - Multi-language text recognition
    - Form and table extraction from images
    """
    
    def __init__(self, llm=None):
        """
        Initialize OCR and Image Agent with dual-format support.
        
        Args:
            llm: Optional LLM instance for vision analysis (e.g., Claude, GPT-4o, Gemini)
        """
        self.agent_name = "ocr_image_agent"
        self.supported_operations = [
            "ocr_image",
            "extract_images_from_pdf",
            "extract_images_from_doc",
            "process_screenshot",
            "batch_ocr",
            "detect_language",
            "extract_table_from_image",
            "analyze_visual_content"  # NEW: Multimodal vision analysis
        ]
        self.llm = llm
        
        # Initialize event bus for event-driven workflows
        self.event_bus = get_event_bus()
        
        logger.info("OCR and Image Agent initialized with dual-format support")
        self._check_dependencies()
        self._initialize_vision_capabilities()
    
    
    def _check_dependencies(self):
        """Check if required OCR and image libraries are installed."""
        self.ocr_engines = []
        
        # Check for Tesseract OCR
        try:
            import pytesseract
            from PIL import Image
            # Try to get version to verify tesseract is installed
            pytesseract.get_tesseract_version()
            self.ocr_engines.append("tesseract")
            logger.debug("Tesseract OCR available")
        except Exception:
            logger.debug("Tesseract OCR not available")
        
        # Check for EasyOCR
        try:
            import easyocr
            self.ocr_engines.append("easyocr")
            logger.debug("EasyOCR available")
        except ImportError:
            logger.debug("EasyOCR not available")
        
        # Check for PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.ocr_engines.append("paddleocr")
            logger.debug("PaddleOCR available")
        except ImportError:
            logger.debug("PaddleOCR not available")
        
        # Check for PIL/Pillow
        try:
            from PIL import Image
            self.has_pil = True
            logger.debug("PIL/Pillow available")
        except ImportError:
            self.has_pil = False
            logger.warning("PIL/Pillow not available")
        
        # Check for pdf2image
        try:
            import pdf2image
            self.has_pdf2image = True
            logger.debug("pdf2image available")
        except ImportError:
            self.has_pdf2image = False
            logger.debug("pdf2image not available")
        
        if not self.ocr_engines:
            logger.warning(
                "No OCR engine found. Install one of: "
                "pip install pytesseract pillow (and install tesseract-ocr), "
                "pip install easyocr, "
                "pip install paddlepaddle paddleocr"
            )
        
        self.preferred_engine = self.ocr_engines[0] if self.ocr_engines else None
        if self.preferred_engine:
            logger.info(f"Using {self.preferred_engine} as primary OCR engine")
    
    
    def _initialize_vision_capabilities(self):
        """Initialize vision analysis capabilities."""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Check if vision analysis is enabled
        self.vision_analysis_enabled = os.getenv('ENABLE_VISION_ANALYSIS', 'true').lower() == 'true'
        
        # Get vision provider (falls back to agent LLM provider)
        self.vision_provider = os.getenv('VISION_LLM_PROVIDER') or os.getenv('AGENT_LLM_PROVIDER', 'google')
        
        # Check for vision-capable models
        self.vision_model = os.getenv('VISION_LLM_MODEL')
        self.auto_detect_charts = os.getenv('AUTO_DETECT_CHARTS', 'true').lower() == 'true'
        
        if self.vision_analysis_enabled:
            logger.info(f"Vision analysis enabled with provider: {self.vision_provider}")
        else:
            logger.info("Vision analysis disabled (ENABLE_VISION_ANALYSIS=false)")
    
    
    
    def ocr_image(
        self,
        image_path: str,
        language: str = 'eng',
        engine: Optional[str] = None,
        preprocessing: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            language: Language code (e.g., 'eng', 'spa', 'fra', 'chi_sim')
            engine: OCR engine to use ('tesseract', 'easyocr', 'paddleocr')
            preprocessing: Whether to apply image preprocessing
        
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            if not self.ocr_engines:
                return {
                    "success": False,
                    "error": "No OCR engine installed",
                    "file": image_path
                }
            
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}",
                    "file": str(image_path_obj)
                }
            
            # Select engine
            selected_engine = engine or self.preferred_engine
            if selected_engine not in self.ocr_engines:
                selected_engine = self.preferred_engine
            
            logger.info(f"OCR processing: {image_path} using {selected_engine}")
            
            # Process with selected engine
            if selected_engine == "tesseract":
                return self._ocr_with_tesseract(image_path_obj, language, preprocessing)
            
            elif selected_engine == "easyocr":
                return self._ocr_with_easyocr(image_path_obj, language)
            
            elif selected_engine == "paddleocr":
                return self._ocr_with_paddleocr(image_path_obj, language)
            
            return {
                "success": False,
                "error": f"OCR engine {selected_engine} not available",
                "file": str(image_path_obj)
            }
        
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(image_path)
            }
    
    
    def _ocr_with_tesseract(
        self,
        image_path: Path,
        language: str,
        preprocessing: bool
    ) -> Dict[str, Any]:
        """Process image with Tesseract OCR."""
        try:
            import pytesseract
            from PIL import Image, ImageEnhance, ImageFilter
            
            # Open image
            image = Image.open(str(image_path))
            
            # Apply preprocessing if enabled
            if preprocessing:
                # Convert to grayscale
                image = image.convert('L')
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2)
                
                # Apply slight sharpening
                image = image.filter(ImageFilter.SHARPEN)
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=language)
            
            # Get detailed data
            data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)
            
            # Calculate confidence
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            logger.info(f"Tesseract OCR completed: {len(text)} characters, {avg_confidence:.1f}% confidence")
            
            return {
                "success": True,
                "file": str(image_path),
                "engine": "tesseract",
                "text": text,
                "language": language,
                "confidence": avg_confidence,
                "character_count": len(text),
                "word_count": len(text.split()),
                "line_count": len(text.splitlines()),
                "detailed_data": data,
                "processed_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Tesseract OCR error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(image_path),
                "engine": "tesseract"
            }
    
    
    def _ocr_with_easyocr(
        self,
        image_path: Path,
        language: str
    ) -> Dict[str, Any]:
        """Process image with EasyOCR."""
        try:
            import easyocr
            
            # Convert language code
            lang_map = {
                'eng': 'en',
                'spa': 'es',
                'fra': 'fr',
                'deu': 'de',
                'chi_sim': 'ch_sim',
                'chi_tra': 'ch_tra',
                'jpn': 'ja',
                'kor': 'ko'
            }
            lang_code = lang_map.get(language, 'en')
            
            # Initialize reader
            reader = easyocr.Reader([lang_code], gpu=False)
            
            # Read text
            results = reader.readtext(str(image_path))
            
            # Combine text and calculate confidence
            text_lines = []
            confidences = []
            
            for bbox, text, conf in results:
                text_lines.append(text)
                confidences.append(conf * 100)
            
            full_text = '\n'.join(text_lines)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            logger.info(f"EasyOCR completed: {len(full_text)} characters, {avg_confidence:.1f}% confidence")
            
            return {
                "success": True,
                "file": str(image_path),
                "engine": "easyocr",
                "text": full_text,
                "language": language,
                "confidence": avg_confidence,
                "character_count": len(full_text),
                "word_count": len(full_text.split()),
                "line_count": len(text_lines),
                "detailed_results": [
                    {
                        "text": text,
                        "confidence": conf * 100,
                        "bbox": bbox
                    }
                    for bbox, text, conf in results
                ],
                "processed_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"EasyOCR error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(image_path),
                "engine": "easyocr"
            }
    
    
    def _ocr_with_paddleocr(
        self,
        image_path: Path,
        language: str
    ) -> Dict[str, Any]:
        """Process image with PaddleOCR."""
        try:
            from paddleocr import PaddleOCR
            
            # Convert language code
            lang_map = {
                'eng': 'en',
                'spa': 'es',
                'fra': 'french',
                'deu': 'german',
                'chi_sim': 'ch',
                'chi_tra': 'chinese_cht',
                'jpn': 'japan',
                'kor': 'korean'
            }
            lang_code = lang_map.get(language, 'en')
            
            # Initialize OCR
            ocr = PaddleOCR(use_angle_cls=True, lang=lang_code, show_log=False)
            
            # Process image
            results = ocr.ocr(str(image_path), cls=True)
            
            # Extract text and confidence
            text_lines = []
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        text = line[1][0]
                        conf = line[1][1] * 100
                        text_lines.append(text)
                        confidences.append(conf)
            
            full_text = '\n'.join(text_lines)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            logger.info(f"PaddleOCR completed: {len(full_text)} characters, {avg_confidence:.1f}% confidence")
            
            return {
                "success": True,
                "file": str(image_path),
                "engine": "paddleocr",
                "text": full_text,
                "language": language,
                "confidence": avg_confidence,
                "character_count": len(full_text),
                "word_count": len(full_text.split()),
                "line_count": len(text_lines),
                "processed_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"PaddleOCR error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(image_path),
                "engine": "paddleocr"
            }
    
    
    def extract_images_from_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        page_range: Optional[tuple] = None,
        dpi: int = 300
    ) -> Dict[str, Any]:
        """
        Extract images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            page_range: Tuple of (start_page, end_page) to extract
            dpi: Resolution for image extraction
        
        Returns:
            Dictionary with extraction results
        """
        try:
            pdf_path_obj = Path(pdf_path)
            if not pdf_path_obj.exists():
                return {
                    "success": False,
                    "error": f"PDF file not found: {pdf_path}",
                    "file": str(pdf_path_obj)
                }
            
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Extracting images from PDF: {pdf_path}")
            
            # Method 1: Extract embedded images using PyPDF2/pypdf
            try:
                embedded_images = self._extract_embedded_images_from_pdf(
                    pdf_path_obj, output_dir_obj, page_range
                )
            except Exception as e:
                logger.warning(f"Failed to extract embedded images: {str(e)}")
                embedded_images = []
            
            # Method 2: Convert PDF pages to images using pdf2image
            page_images = []
            if self.has_pdf2image:
                try:
                    page_images = self._convert_pdf_pages_to_images(
                        pdf_path_obj, output_dir_obj, page_range, dpi
                    )
                except Exception as e:
                    logger.warning(f"Failed to convert pages to images: {str(e)}")
            
            total_images = len(embedded_images) + len(page_images)
            
            logger.info(f"Extracted {total_images} images from PDF")
            
            return {
                "success": True,
                "file": str(pdf_path),
                "output_dir": str(output_dir_obj),
                "embedded_images": embedded_images,
                "page_images": page_images,
                "total_images": total_images,
                "extracted_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(pdf_path)
            }
    
    
    def _extract_embedded_images_from_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
        page_range: Optional[tuple]
    ) -> List[str]:
        """Extract embedded images from PDF."""
        extracted_images = []
        
        try:
            import PyPDF2
            from PIL import Image
            import io
            
            pdf_reader = PyPDF2.PdfReader(str(pdf_path))
            num_pages = len(pdf_reader.pages)
            
            start = page_range[0] if page_range else 0
            end = page_range[1] if page_range else num_pages
            
            for page_num in range(start, min(end, num_pages)):
                page = pdf_reader.pages[page_num]
                
                # Check if page has resources with XObject
                if '/Resources' in page and isinstance(page['/Resources'], dict):
                    resources = page['/Resources']
                    if '/XObject' in resources:  # type: ignore
                        xobjects = resources['/XObject'].get_object()  # type: ignore
                    
                    for obj_name in xobjects:
                        obj = xobjects[obj_name]
                        
                        if obj['/Subtype'] == '/Image':
                            try:
                                # Extract image data
                                if '/Filter' in obj:
                                    if obj['/Filter'] == '/DCTDecode':
                                        # JPEG image
                                        img_path = output_dir / f"page{page_num+1}_{obj_name[1:]}.jpg"
                                        with open(img_path, 'wb') as img_file:
                                            img_file.write(obj.get_data())
                                        extracted_images.append(str(img_path))
                                    
                                    elif obj['/Filter'] == '/FlateDecode':
                                        # PNG-like image
                                        img_data = obj.get_data()
                                        img = Image.frombytes(
                                            'RGB',
                                            (obj['/Width'], obj['/Height']),
                                            img_data
                                        )
                                        img_path = output_dir / f"page{page_num+1}_{obj_name[1:]}.png"
                                        img.save(img_path)
                                        extracted_images.append(str(img_path))
                            
                            except Exception as e:
                                logger.debug(f"Could not extract image {obj_name}: {str(e)}")
                                continue
        
        except Exception as e:
            logger.warning(f"Embedded image extraction failed: {str(e)}")
        
        return extracted_images
    
    
    def _convert_pdf_pages_to_images(
        self,
        pdf_path: Path,
        output_dir: Path,
        page_range: Optional[tuple],
        dpi: int
    ) -> List[str]:
        """Convert PDF pages to images."""
        page_images = []
        
        try:
            from pdf2image import convert_from_path
            
            # Convert pages
            if page_range:
                first_page = page_range[0] + 1
                last_page = page_range[1]
                images = convert_from_path(
                    str(pdf_path),
                    dpi=dpi,
                    first_page=first_page,
                    last_page=last_page
                )
            else:
                images = convert_from_path(str(pdf_path), dpi=dpi)
            
            # Save images
            for i, image in enumerate(images):
                page_num = (page_range[0] if page_range else 0) + i + 1
                img_path = output_dir / f"page_{page_num}.png"
                image.save(img_path, 'PNG')
                page_images.append(str(img_path))
        
        except Exception as e:
            logger.warning(f"PDF to image conversion failed: {str(e)}")
        
        return page_images
    
    
    def batch_ocr(
        self,
        image_paths: List[str],
        language: str = 'eng',
        engine: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform OCR on multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            language: Language code for OCR
            engine: OCR engine to use
        
        Returns:
            Dictionary with batch OCR results
        """
        try:
            logger.info(f"Batch OCR processing {len(image_paths)} images")
            
            results = []
            success_count = 0
            failed_count = 0
            
            for image_path in image_paths:
                result = self.ocr_image(image_path, language, engine, preprocessing=True)
                results.append(result)
                
                if result.get('success'):
                    success_count += 1
                else:
                    failed_count += 1
            
            logger.info(f"Batch OCR completed: {success_count} succeeded, {failed_count} failed")
            
            return {
                "success": True,
                "total_images": len(image_paths),
                "successful": success_count,
                "failed": failed_count,
                "results": results,
                "processed_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error in batch OCR: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_images": len(image_paths)
            }
    
    
    def _detect_visual_content_type(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if image contains charts, graphs, diagrams, or other complex visual content.
        
        This enables automatic routing to vision analysis for deeper interpretation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with detection results and content type classification
        """
        try:
            from PIL import Image
            import numpy as np
            
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return {
                    "detected": False,
                    "error": f"Image not found: {image_path}",
                    "content_types": []
                }
            
            # Open image
            image = Image.open(str(image_path_obj))
            image_array = np.array(image)
            
            # Analyze image characteristics
            detected_types = []
            
            # Check for typical chart/graph indicators
            # 1. Look for axis-like patterns (lines)
            if len(image_array.shape) >= 2:
                height, width = image_array.shape[:2]
                
                # Detect lines (potential axes)
                has_lines = self._detect_lines(image_array)
                if has_lines:
                    detected_types.append("chart_or_graph")
                
                # Detect color gradients (heatmaps, color scales)
                has_gradients = self._detect_color_gradients(image_array)
                if has_gradients:
                    detected_types.append("heatmap_or_color_scale")
                
                # Detect structured layouts (tables, forms)
                has_structure = self._detect_structured_layout(image_array)
                if has_structure:
                    detected_types.append("table_or_form")
                
                # Detect diagrams (boxes, shapes, connections)
                has_shapes = self._detect_shapes(image_array)
                if has_shapes:
                    detected_types.append("diagram_or_flowchart")
            
            return {
                "detected": len(detected_types) > 0,
                "content_types": detected_types,
                "image_path": str(image_path_obj),
                "image_size": (image.width, image.height),
                "recommendation": "use_vision_analysis" if detected_types else "use_standard_ocr"
            }
        
        except Exception as e:
            logger.warning(f"Error detecting visual content type: {str(e)}")
            return {
                "detected": False,
                "error": str(e),
                "content_types": [],
                "recommendation": "use_standard_ocr"
            }
    
    
    def _detect_lines(self, image_array: np.ndarray) -> bool:
        """Detect if image contains line patterns (potential chart axes)."""
        try:
            import cv2
            
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array.astype(np.uint8)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for straight lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
            
            return lines is not None and len(lines) > 3
        except ImportError:
            # OpenCV not available, use basic detection
            return False
    
    
    def _detect_color_gradients(self, image_array: np.ndarray) -> bool:
        """Detect color gradients indicating heatmaps or color scales."""
        try:
            import cv2
            
            # Only for color images
            if len(image_array.shape) != 3:
                return False
            
            # Calculate color variance
            color_variance = np.var(image_array, axis=(0, 1))
            avg_variance = np.mean(color_variance)
            
            # High color variance suggests gradients/heatmaps
            return avg_variance > 1000
        except:
            return False
    
    
    def _detect_structured_layout(self, image_array: np.ndarray) -> bool:
        """Detect structured layouts like tables or forms."""
        try:
            import cv2
            
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array.astype(np.uint8)
            
            # Look for grid patterns
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # High number of rectangular contours suggests table/form
            rectangular_count = sum(1 for c in contours if 0.7 < cv2.contourArea(c) / (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]) < 1.0)
            
            return rectangular_count > 5
        except (ImportError, Exception):
            return False
    
    
    def _detect_shapes(self, image_array: np.ndarray) -> bool:
        """Detect shapes and objects indicating diagrams."""
        try:
            import cv2
            
            # Convert to grayscale
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array.astype(np.uint8)
            
            # Find circles and contours
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30)
            
            return circles is not None and len(circles[0]) > 1
        except (ImportError, Exception):
            return False
    
    
    def analyze_visual_content(
        self,
        image_path: str,
        analysis_prompt: Optional[str] = None,
        auto_detect: bool = True
    ) -> Dict[str, Any]:
        """
        Use multimodal vision LLM to analyze complex visual content like charts, diagrams, and images.
        
        This provides deeper interpretation beyond standard OCR text extraction.
        
        Args:
            image_path: Path to the image file
            analysis_prompt: Custom prompt for analysis (e.g., "Describe this chart's trends")
            auto_detect: If True, automatically detect content type and suggest analysis
            
        Returns:
            Dictionary with vision analysis results and extracted insights
        """
        try:
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}",
                    "file": str(image_path_obj)
                }
            
            # Check if vision analysis is enabled
            if not self.vision_analysis_enabled:
                logger.warning("Vision analysis is disabled. Enable with ENABLE_VISION_ANALYSIS=true")
                return {
                    "success": False,
                    "error": "Vision analysis is disabled",
                    "recommendation": "Set ENABLE_VISION_ANALYSIS=true in .env"
                }
            
            # Auto-detect content type if enabled
            content_detection = None
            if auto_detect:
                content_detection = self._detect_visual_content_type(str(image_path_obj))
                logger.info(f"Content detection: {content_detection['content_types']}")
            
            # Initialize vision LLM
            vision_llm = self._initialize_vision_llm()
            if not vision_llm:
                return {
                    "success": False,
                    "error": "Could not initialize vision LLM. Check API keys and configuration.",
                    "provider": self.vision_provider
                }
            
            # Prepare image for LLM
            image_data = self._prepare_image_for_llm(str(image_path_obj))
            if not image_data:
                return {
                    "success": False,
                    "error": "Could not prepare image for vision analysis"
                }
            
            # Build analysis prompt
            if not analysis_prompt:
                # Auto-generate prompt based on detected content
                if content_detection and content_detection['detected']:
                    analysis_prompt = self._generate_analysis_prompt(content_detection['content_types'])
                else:
                    analysis_prompt = "Provide a detailed analysis of this image. Describe any charts, graphs, tables, diagrams, or other visual elements. Extract key information and insights."
            
            logger.info(f"Vision analysis with prompt: {analysis_prompt[:100]}...")
            
            # Call vision LLM
            analysis_result = vision_llm.analyze_image(
                image_data=image_data,
                prompt=analysis_prompt
            )
            
            return {
                "success": True,
                "file": str(image_path_obj),
                "analysis": analysis_result.get('text', ''),
                "content_detected": content_detection['content_types'] if content_detection else [],
                "prompt_used": analysis_prompt,
                "provider": self.vision_provider,
                "metadata": {
                    "detection_results": content_detection,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        except Exception as e:
            logger.error(f"Error in vision analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(image_path)
            }
    
    
    def _initialize_vision_llm(self):
        """
        Initialize the vision-capable LLM based on configuration.
        
        Supports: Google Gemini Pro Vision (native SDK or LangChain), OpenAI GPT-4o, Anthropic Claude 3
        """
        try:
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            
            if self.llm:
                return self.llm
            
            provider = self.vision_provider
            
            if provider == 'google' or provider == 'gemini':
                # Check if using native SDK - use generic env variable
                use_native = os.getenv('USE_NATIVE_SDK', os.getenv('USE_NATIVE_GEMINI_SDK', '')).lower() == 'true'
                
                if use_native:
                    try:
                        from task_manager.utils.llm_client import LLMClient
                        # Try generic LLM_API_KEY first, fall back to GOOGLE_API_KEY
                        api_key = os.getenv('LLM_API_KEY') or os.getenv('GOOGLE_API_KEY')
                        if not api_key:
                            logger.error("LLM_API_KEY or GOOGLE_API_KEY not found in environment")
                            return None
                        
                        model = self.vision_model or 'gemini-2.5-pro-vision'
                        llm = LLMClient(
                            provider='google',
                            model=model,
                            temperature=float(os.getenv('AGENT_LLM_TEMPERATURE', '0.2')),
                            api_version=os.getenv('LLM_API_VERSION', os.getenv('GEMINI_API_VERSION', 'v1alpha')),
                            use_native_sdk=True,
                            api_base_url=os.getenv('LLM_API_BASE_URL') or os.getenv('GEMINI_API_BASE_URL'),
                            api_endpoint_path=os.getenv('LLM_API_ENDPOINT_PATH') or os.getenv('GEMINI_API_ENDPOINT_PATH')
                        )
                        
                        # Wrap with vision capability
                        return VisionLLMWrapper(llm, provider='google')
                    
                    except ImportError as e:
                        logger.warning(f"Native SDK not available: {str(e)}. Falling back to LangChain wrapper.")
                        use_native = False
                
                if not use_native:
                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        api_key = os.getenv('GOOGLE_API_KEY')
                        if not api_key:
                            logger.error("GOOGLE_API_KEY not found in environment")
                            return None
                        
                        model = self.vision_model or 'gemini-2.5-pro-vision'
                        llm = ChatGoogleGenerativeAI(
                            model=model,
                            google_api_key=api_key,
                            temperature=float(os.getenv('AGENT_LLM_TEMPERATURE', '0.2'))
                        )
                        
                        # Wrap with vision capability
                        return VisionLLMWrapper(llm, provider='google')
                    
                    except ImportError:
                        logger.error("langchain-google-genai not installed")
                        return None
            
            elif provider == 'openai':
                try:
                    from langchain_openai import ChatOpenAI
                    api_key = os.getenv('OPENAI_API_KEY')
                    if not api_key:
                        logger.error("OPENAI_API_KEY not found in environment")
                        return None
                    
                    model = self.vision_model or 'gpt-4o'
                    llm = ChatOpenAI(
                        model=model,
                        api_key=api_key,  # type: ignore
                        temperature=float(os.getenv('AGENT_LLM_TEMPERATURE', '0.2'))
                    )
                    
                    return VisionLLMWrapper(llm, provider='openai')
                
                except ImportError:
                    logger.error("langchain-openai not installed")
                    return None
            
            elif provider == 'anthropic':
                try:
                    from langchain_anthropic import ChatAnthropic
                    api_key = os.getenv('ANTHROPIC_API_KEY')
                    if not api_key:
                        logger.error("ANTHROPIC_API_KEY not found in environment")
                        return None
                    
                    model = self.vision_model or 'claude-3-5-sonnet-20241022'
                    llm = ChatAnthropic(  # type: ignore
                        model_name=model,
                        api_key=api_key,  # type: ignore
                        temperature=float(os.getenv('AGENT_LLM_TEMPERATURE', '0.2'))
                    )
                    
                    return VisionLLMWrapper(llm, provider='anthropic')
                
                except ImportError:
                    logger.error("langchain-anthropic not installed")
                    return None
            
            else:
                logger.error(f"Unsupported vision provider: {provider}")
                return None
        
        except Exception as e:
            logger.error(f"Error initializing vision LLM: {str(e)}")
            return None
    
    
    def _prepare_image_for_llm(self, image_path: str) -> Optional[str]:
        """
        Prepare image for LLM by converting to base64 or appropriate format.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image or None if conversion fails
        """
        try:
            # Read image file
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            return image_data
        
        except Exception as e:
            logger.error(f"Error preparing image for LLM: {str(e)}")
            return None
    
    
    def _generate_analysis_prompt(self, content_types: List[str]) -> str:
        """
        Generate contextual analysis prompt based on detected content types.
        
        Args:
            content_types: List of detected content types
            
        Returns:
            Tailored analysis prompt
        """
        prompts = {
            'chart_or_graph': "Analyze this chart/graph. Describe: 1) The type of chart, 2) Axes and units, 3) Key trends and patterns, 4) Notable data points, 5) Overall insights.",
            'heatmap_or_color_scale': "Analyze this heatmap/color scale visualization. Describe: 1) What variable is being represented, 2) The color scale meaning, 3) Spatial patterns, 4) Hotspots or anomalies.",
            'table_or_form': "Analyze this table/form. Extract and describe: 1) Column/row headers, 2) Data structure, 3) Key numerical values, 4) Any totals or summaries.",
            'diagram_or_flowchart': "Analyze this diagram/flowchart. Describe: 1) Main components/boxes, 2) Connections and flow direction, 3) Process or hierarchy shown, 4) Key decision points."
        }
        
        # Combine prompts for multiple content types
        relevant_prompts = [prompts.get(ct, "") for ct in content_types if ct in prompts]
        
        if relevant_prompts:
            combined = " ".join(relevant_prompts)
            return combined
        else:
            return "Provide a detailed analysis of this image. Describe all visible elements, patterns, and information."
    
    
    def analyze_with_fallback(
        self,
        image_path: str,
        analysis_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze image with vision LLM, falling back to standard OCR if vision fails.
        
        This ensures robust handling when vision analysis is unavailable.
        
        Args:
            image_path: Path to the image file
            analysis_prompt: Custom analysis prompt
            
        Returns:
            Dictionary with analysis results (vision or OCR)
        """
        # Try vision analysis first
        if self.vision_analysis_enabled:
            vision_result = self.analyze_visual_content(
                image_path=image_path,
                analysis_prompt=analysis_prompt,
                auto_detect=True
            )
            
            if vision_result.get('success'):
                return vision_result
            else:
                logger.warning(f"Vision analysis failed: {vision_result.get('error')}. Falling back to standard OCR.")
        
        # Fall back to standard OCR
        logger.info(f"Using standard OCR for: {image_path}")
        ocr_result = self.ocr_image(
            image_path=image_path,
            language='eng',
            preprocessing=True
        )
        
        # Enhance OCR result to include fallback indicator
        ocr_result['method'] = 'ocr_fallback'
        ocr_result['vision_available'] = self.vision_analysis_enabled
        
        return ocr_result
    
    def execute_task(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AgentExecutionResponse]:
        """
        Execute an OCR/Image operation with dual-format support.
        
        Supports three calling conventions:
        1. Legacy positional: execute_task(operation, parameters)
        2. Legacy dict: execute_task({'operation': ..., 'parameters': ...})
        3. Standardized: execute_task(AgentExecutionRequest)
        
        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        
        Returns:
            Legacy dict OR AgentExecutionResponse based on input format
        """
        start_time = time.time()
        return_legacy = True
        operation = None
        parameters = None
        task_dict = None
        
        # Detect calling convention
        # Positional arguments (operation, parameters)
        if len(args) == 2:
            operation, parameters = args
            task_dict = {"operation": operation, "parameters": parameters}
            return_legacy = True
            logger.debug("Legacy positional call")
        
        # Single dict argument
        elif len(args) == 1 and isinstance(args[0], dict):
            task_dict = args[0]
            # Check if standardized request (has task_id and task_description)
            if "task_id" in task_dict and "task_description" in task_dict:
                return_legacy = False
                logger.debug(f"Standardized request call: task_id={task_dict.get('task_id')}")
            else:
                return_legacy = True
                logger.debug("Legacy dict call")
            operation = task_dict.get("operation")
            parameters = task_dict.get("parameters", {})
        
        # Keyword arguments (operation=..., parameters=...)
        elif "operation" in kwargs:
            operation = kwargs.get("operation")
            parameters = kwargs.get("parameters", {})
            task_dict = {"operation": operation, "parameters": parameters}
            return_legacy = True
            logger.debug("Legacy keyword call")
        
        else:
            raise InvalidParameterError(
                parameter_name="task",
                message="Invalid call to execute_task. Use one of:\n"
                "  - execute_task(operation, parameters)\n"
                "  - execute_task({'operation': ..., 'parameters': ...})\n"
                "  - execute_task(AgentExecutionRequest)"
            )
        
        try:
            task_id = task_dict.get("task_id", f"ocr_{int(time.time())}")  # type: ignore
            
            # Ensure parameters is not None
            if parameters is None:
                parameters = {}
            
            # Ensure operation is not None
            if operation is None:
                operation = "unknown"
            
            logger.info(f"Executing OCR/Image operation: {operation} (task_id={task_id})")
            
            # Execute the operation using existing methods
            if operation == "ocr_image":
                result = self.ocr_image(
                    image_path=parameters.get('image_path', ''),
                    language=parameters.get('language', 'eng'),
                    engine=parameters.get('engine'),
                    preprocessing=parameters.get('preprocessing', True)
                )
            
            elif operation == "extract_images_from_pdf":
                result = self.extract_images_from_pdf(
                    pdf_path=parameters.get('pdf_path', ''),
                    output_dir=parameters.get('output_dir', ''),
                    page_range=parameters.get('page_range'),
                    dpi=parameters.get('dpi', 300)
                )
            
            elif operation == "batch_ocr":
                result = self.batch_ocr(
                    image_paths=parameters.get('image_paths', []),
                    language=parameters.get('language', 'eng'),
                    engine=parameters.get('engine')
                )
            
            elif operation == "process_screenshot":
                # Alias for ocr_image
                result = self.ocr_image(
                    image_path=parameters.get('image_path', ''),
                    language=parameters.get('language', 'eng'),
                    engine=parameters.get('engine'),
                    preprocessing=parameters.get('preprocessing', True)
                )
            
            elif operation == "analyze_visual_content":
                result = self.analyze_visual_content(
                    image_path=parameters.get('image_path', ''),
                    analysis_prompt=parameters.get('analysis_prompt'),
                    auto_detect=parameters.get('auto_detect', True)
                )
            
            else:
                result = {
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                    "supported_operations": self.supported_operations
                }
            
            # Convert legacy result to standardized response
            standard_response = self._convert_to_standard_response(
                result,
                operation,
                task_id,
                start_time
            )
            
            # Publish completion event for event-driven workflows
            self._publish_completion_event(task_id, operation, standard_response)
            
            # Return in requested format
            if return_legacy:
                # Convert back to legacy format for backward compatibility
                return self._convert_to_legacy_response(standard_response)
            else:
                return standard_response
        
        except Exception as e:
            logger.error(f"Error executing OCR/Image task: {e}", exc_info=True)
            
            # Create standardized error response
            error = exception_to_error_response(
                e,
                source=self.agent_name,
                task_id=task_dict.get("task_id", "unknown") if task_dict else "unknown"
            )
            
            error_response: AgentExecutionResponse = {
                "status": "failure",
                "success": False,
                "result": {},
                "artifacts": [],
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.agent_name,
                "operation": operation or "unknown",
                "blackboard_entries": [],
                "warnings": []
            }
            # Add error field separately to handle TypedDict
            error_response["error"] = error  # type: ignore
            
            if return_legacy:
                return self._convert_to_legacy_response(error_response)
            else:
                return error_response
    
    # Backward compatibility: keep run_operation as alias
    def run_operation(
        self,
        operation: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an OCR/Image operation (backward compatibility wrapper).
        
        DEPRECATED: Use execute_task() instead.
        This method is maintained for backward compatibility only.
        
        Args:
            operation: Type of operation
            parameters: Operation parameters
        
        Returns:
            Result dictionary
        """
        logger.debug("run_operation() called - forwarding to execute_task()")
        return self.execute_task(operation, parameters)  # type: ignore
    
    def _convert_to_standard_response(
        self,
        legacy_result: Dict[str, Any],
        operation: str,
        task_id: str,
        start_time: float
    ) -> AgentExecutionResponse:
        """Convert legacy result dict to standardized AgentExecutionResponse."""
        success = legacy_result.get("success", False)
        
        # Extract artifacts from result
        artifacts = []
        
        # Handle image output files
        if "output_images" in legacy_result and success:
            for img_path in legacy_result.get("output_images", []):
                if Path(img_path).exists():
                    artifacts.append({
                        "type": "image",
                        "path": str(img_path),
                        "size_bytes": Path(img_path).stat().st_size,
                        "description": f"Extracted image from {operation} operation"
                    })
        
        # Handle text output files
        if "output_file" in legacy_result and success:
            output_file = legacy_result["output_file"]
            if Path(output_file).exists():
                artifacts.append({
                    "type": "txt",
                    "path": str(output_file),
                    "size_bytes": Path(output_file).stat().st_size,
                    "description": f"OCR text output from {operation} operation"
                })
        
        # Create blackboard entries for sharing data
        blackboard_entries = []
        if success and "text" in legacy_result:
            blackboard_entries.append({
                "key": f"ocr_text_{task_id}",
                "value": legacy_result.get("text", ""),
                "scope": "workflow",
                "ttl_seconds": 3600
            })
        
        if success and "table_data" in legacy_result:
            blackboard_entries.append({
                "key": f"table_data_{task_id}",
                "value": legacy_result.get("table_data", []),
                "scope": "workflow",
                "ttl_seconds": 3600
            })
        
        # Build standardized response
        response: AgentExecutionResponse = {
            "status": "success" if success else "failure",
            "success": success,
            "result": {
                k: v for k, v in legacy_result.items()
                if k not in ["success", "output_images", "output_file"]
            },
            "artifacts": artifacts,
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "operation": operation,
            "blackboard_entries": blackboard_entries,
            "warnings": []
        }
        
        # Add error field if present (handle TypedDict)
        if not success and "error" in legacy_result:
            response["error"] = create_error_response(  # type: ignore
                error_code="OCR_001",
                error_type="execution_error",
                message=legacy_result.get("error", "Unknown error"),
                source=self.agent_name
            )
        
        return response
    
    def _convert_to_legacy_response(self, standard_response: AgentExecutionResponse) -> Dict[str, Any]:
        """Convert standardized response back to legacy format for backward compatibility."""
        legacy: Dict[str, Any] = {
            "success": standard_response["success"],
        }
        
        # Add output_images from artifacts
        image_artifacts = [a for a in standard_response["artifacts"] if a["type"] == "image"]
        if image_artifacts:
            legacy["output_images"] = [a["path"] for a in image_artifacts]
        
        # Add output_file from artifacts
        text_artifacts = [a for a in standard_response["artifacts"] if a["type"] == "txt"]
        if text_artifacts:
            legacy["output_file"] = text_artifacts[0]["path"]
        
        # Add error if present (use .get() for NotRequired field)
        error = standard_response.get("error")  # type: ignore
        if error:
            legacy["error"] = error["message"]  # type: ignore
        
        # Merge result fields into top level (legacy pattern)
        if isinstance(standard_response["result"], dict):
            for key, value in standard_response["result"].items():
                if key not in legacy:
                    legacy[key] = value
        
        return legacy
    
    def _publish_completion_event(
        self,
        task_id: str,
        operation: str,
        response: AgentExecutionResponse
    ):
        """Publish task completion event for event-driven workflows."""
        try:
            # Choose event type based on operation
            if operation == "ocr_image" or operation == "process_screenshot":
                event_type = "ocr_extraction_completed"
            elif "table" in operation:
                event_type = "table_data_ready"
            elif "extract_images" in operation:
                event_type = "images_extracted"
            else:
                event_type = "ocr_operation_completed"
            
            event = create_system_event(
                event_type=event_type,
                event_category="task_lifecycle",
                source_agent=self.agent_name,
                payload={
                    "task_id": task_id,
                    "operation": operation,
                    "success": response["success"],
                    "artifacts": response["artifacts"],
                    "blackboard_keys": [entry["key"] for entry in response["blackboard_entries"]]
                }
            )
            self.event_bus.publish(event)
            logger.debug(f"Published completion event for task {task_id}")
        except Exception as e:
            logger.warning(f"Failed to publish completion event: {e}")


class VisionLLMWrapper:
    """
    Wrapper for vision-capable LLMs to provide a unified interface.
    
    Supports multiple providers: Google Gemini, OpenAI GPT-4o, Anthropic Claude
    """
    
    def __init__(self, llm, provider: str):
        """
        Initialize vision LLM wrapper.
        
        Args:
            llm: LangChain LLM instance (ChatGoogleGenerativeAI, ChatOpenAI, etc.)
            provider: LLM provider name ('google', 'openai', 'anthropic')
        """
        self.llm = llm
        self.provider = provider
        self.logger = get_logger(__name__)
    
    
    def analyze_image(
        self,
        image_data: str,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze image using vision LLM.
        
        Args:
            image_data: Base64 encoded image data
            prompt: Analysis prompt/question
            
        Returns:
            Dictionary with analysis results
        """
        try:
            from langchain_core.messages import HumanMessage
            
            # Build message with image based on provider
            if self.provider == 'google':
                message = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                )
            
            elif self.provider == 'openai':
                message = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                )
            
            elif self.provider == 'anthropic':
                import base64
                import hashlib
                
                # Anthropic format
                message = HumanMessage(
                    content=[
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                )
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported provider: {self.provider}",
                    "text": ""
                }
            
            # Call LLM
            response = self.llm.invoke([message])
            
            # Extract text from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            self.logger.info(f"Vision analysis completed with {self.provider}")
            
            return {
                "success": True,
                "text": response_text,
                "provider": self.provider
            }
        
        except Exception as e:
            self.logger.error(f"Error in vision analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }