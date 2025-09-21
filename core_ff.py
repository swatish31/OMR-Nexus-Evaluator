# core_omr.py
import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class OMRProcessor:
    def __init__(self):
        self.answer_keys = self.load_answer_keys()
        
    def load_answer_keys(self) -> Dict:
        """Load all answer key versions from JSON files"""
        keys = {}
        try:
            with open('answer_keys/version_1.json', 'r') as f:
                keys['version_1'] = json.load(f)
        except FileNotFoundError:
            print("Warning: Answer key files not found. Using default keys.")
            keys['version_1'] = {
                "PYTHON": ["B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A"],
                "DATA_ANALYSIS": ["C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B"],
                "MySQL": ["D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C"],
                "POWER_BI": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D"],
                "Adv_STATS": ["B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A", "B", "C", "D", "A"]
            }
        return keys

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale and enhance contrast"""
        # Resize if image is too large
        height, width = image.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = 800
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to preserve edges while reducing noise
        blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Use adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 21, 10)
        
        return thresh, image

    def get_manual_bubble_positions(self, height: int, width: int) -> List[Dict]:
        """Get manual bubble positions with optimized parameters"""
        # ADJUST THESE PARAMETERS BASED ON YOUR OMR SHEET:
        start_y = 190    # Y position of first question row
        spacing_y = 28   # Vertical spacing between rows
        
        start_x = 110    # X position of first bubble group
        group_spacing = 118  # Horizontal spacing between question groups
        bubble_spacing = 23  # Horizontal spacing between bubbles in a group
        
        bubble_positions = []
        
        for row in range(20):  # 20 rows
            y = start_y + row * spacing_y
            
            for group in range(5):  # 5 questions per row
                x_base = start_x + group * group_spacing
                
                for bubble in range(4):  # 4 options per question
                    x = x_base + bubble * bubble_spacing
                    
                    if 0 <= x < width and 0 <= y < height:
                        bubble_positions.append({
                            'x': x,
                            'y': y,
                            'question': row * 5 + group + 1,
                            'option': chr(65 + bubble)
                        })
        
        return bubble_positions

    def analyze_bubble_simple(self, thresh: np.ndarray, x: int, y: int, radius: int = 8) -> Dict:
        """Simple bubble analysis"""
        y1, y2 = max(0, y-radius), min(thresh.shape[0], y+radius)
        x1, x2 = max(0, x-radius), min(thresh.shape[1], x+radius)
        
        if x1 >= x2 or y1 >= y2:
            return {'fill_percentage': 0, 'is_filled': False}
        
        roi = thresh[y1:y2, x1:x2]
        fill_percentage = (cv2.countNonZero(roi) / roi.size) * 100
        
        return {
            'fill_percentage': fill_percentage,
            'is_filled': False  # Will be set later based on calibrated threshold
        }

    def calibrate_detection(self, thresh: np.ndarray, bubble_positions: List[Dict]) -> float:
        """Calibrate detection threshold based on image statistics"""
        fill_percentages = []
        
        # Sample bubbles to determine optimal threshold
        for i, position in enumerate(bubble_positions[:100]):  # Sample first 100 bubbles
            analysis = self.analyze_bubble_simple(thresh, position['x'], position['y'])
            fill_percentages.append(analysis['fill_percentage'])
        
        if fill_percentages:
            # Use more robust statistical analysis
            fill_percentages = np.array(fill_percentages)
            
            # Remove outliers
            q1 = np.percentile(fill_percentages, 25)
            q3 = np.percentile(fill_percentages, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            filtered_percentages = fill_percentages[(fill_percentages >= lower_bound) & (fill_percentages <= upper_bound)]
            
            if len(filtered_percentages) > 0:
                mean_fill = np.mean(filtered_percentages)
                std_fill = np.std(filtered_percentages)
                
                # Set threshold to mean + 2 standard deviations for better separation
                calibrated_threshold = mean_fill + (2 * std_fill)
                calibrated_threshold = max(25, min(calibrated_threshold, 60))  # Reasonable bounds
                
                print(f"Calibrated threshold: {calibrated_threshold:.1f}% (Mean: {mean_fill:.1f}, Std: {std_fill:.1f})")
                return calibrated_threshold
        
        return 35  # Default threshold

    def extract_answers_improved(self, thresh: np.ndarray, bubble_positions: List[Dict]) -> List[str]:
        """Improved answer extraction with better detection"""
        answers = [''] * 100
        
        # First, calibrate the threshold for this specific image
        optimal_threshold = self.calibrate_detection(thresh, bubble_positions)
        
        # Collect all bubble data using the calibrated threshold
        bubble_data = []
        for position in bubble_positions:
            analysis = self.analyze_bubble_simple(thresh, position['x'], position['y'])
            is_filled = analysis['fill_percentage'] > optimal_threshold
            
            bubble_data.append({
                'question': position['question'] - 1,
                'option': position['option'],
                'fill_pct': analysis['fill_percentage'],
                'is_filled': is_filled
            })
        
        # Process each question
        for question_num in range(100):
            question_bubbles = [b for b in bubble_data if b['question'] == question_num]
            
            if not question_bubbles:
                continue
            
            # Find all filled bubbles
            filled_bubbles = [b for b in question_bubbles if b['is_filled']]
            
            if len(filled_bubbles) == 1:
                # Only one bubble filled - accept it
                answers[question_num] = filled_bubbles[0]['option']
            elif len(filled_bubbles) > 1:
                # Multiple bubbles filled - choose the one with highest fill percentage
                filled_bubbles.sort(key=lambda x: x['fill_pct'], reverse=True)
                if filled_bubbles[0]['fill_pct'] > filled_bubbles[1]['fill_pct'] * 1.2:  # Less strict ratio
                    answers[question_num] = filled_bubbles[0]['option']
                else:
                    answers[question_num] = 'X'  # Ambiguous
            else:
                # No bubbles meet threshold, but check if any are close
                question_bubbles.sort(key=lambda x: x['fill_pct'], reverse=True)
                if question_bubbles[0]['fill_pct'] > optimal_threshold * 0.7:
                    answers[question_num] = question_bubbles[0]['option']
        
        return answers, optimal_threshold

    def calculate_scores(self, answers: List[str], version: str = 'version_1') -> Dict:
        """Calculate subject-wise scores"""
        subject_scores = {}
        key = self.answer_keys[version]
        
        for i, subject in enumerate(["PYTHON", "DATA_ANALYSIS", "MySQL", "POWER_BI", "Adv_STATS"]):
            start_idx = i * 20
            end_idx = start_idx + 20
            
            score = 0
            for j in range(start_idx, end_idx):
                if j < len(answers) and answers[j] == key[subject][j - start_idx]:
                    score += 1
            
            subject_scores[subject] = score
        
        subject_scores['total'] = sum(subject_scores.values())
        return subject_scores

    def process_image(self, image_path: str) -> Dict:
        """Main processing pipeline"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Original image size: {image.shape}")
        
        # Preprocess
        thresh, resized_image = self.preprocess_image(image)
        
        # Get bubble positions
        bubble_positions = self.get_manual_bubble_positions(thresh.shape[0], thresh.shape[1])
        print(f"Using {len(bubble_positions)} bubble positions")
        
        # Extract answers with improved detection
        answers, optimal_threshold = self.extract_answers_improved(thresh, bubble_positions)
        
        # Calculate scores
        scores = self.calculate_scores(answers)
        
        return {
            'image_path': image_path,
            'answers': answers,
            'scores': scores,
            'threshold': optimal_threshold,
            'status': 'processed'
        }