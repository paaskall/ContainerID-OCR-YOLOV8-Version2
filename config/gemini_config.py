import google.generativeai as genai
import os
import cv2
import time
from PIL import Image
import re

class GeminiConfig:
    def __init__(self):
        self.api_key = None
        self.model = None
        self.configured = False
        
    def setup_api_key(self, api_key):
        try:
            genai.configure(api_key=api_key)
            
            try:
                self.model = genai.GenerativeModel('models/gemini-2.5-flash')
                print("Using model: gemini-2.5-flash")
            except Exception as e:
                print(f"gemini-2.5-flash not available: {e}")
                try:
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
                    print("Using model: gemini-1.5-flash")
                except Exception as e2:
                    print(f"gemini-1.5-flash not available: {e2}")
                    try:
                        self.model = genai.GenerativeModel('gemini-1.5-pro')
                        print("Using model: gemini-1.5-pro")
                    except Exception as e3:
                        print(f"All vision models failed: {e3}")
                        return False
            
            self.api_key = api_key
            self.configured = True
            print("Gemini API configured successfully")
            return True
            
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            return False
            
    def get_ocr_prompt(self, label):
        prompts = {
            'Container_ID': """Read the container ID. Format: 4 capital letters followed by 7 numbers.
            Examples: MSCU1234567, APLU7654321, TCLU1234567.
            Return only the container ID or TIDAK_TERBACA.""",
            
            'Container_ID_Vertikal': """Read the vertical container ID. Format: 4 capital letters + 7 numbers.
            The text might be rotated or in vertical orientation.
            Examples: MSCU1234567, APLU7654321.
            Return only the container ID or TIDAK_TERBACA.""",
            
            'truck_id': """Read the truck identification text. This could be:
            - Truck number/unit number
            - License plate number on truck body
            Return the truck ID text or TIDAK_TERBACA.""",
            
            'plate_number': """Read the vehicle license plate number. 
            This is usually a combination of letters and numbers.
            Examples: B 1234 ABC, AB 123 CD, etc.
            Return only the plate number or TIDAK_TERBACA."""
        }
        return prompts.get(label, """Read the text in this image. Return the text or TIDAK_TERBACA.""")
    
    def extract_text_from_image(self, image, label):
        if not self.configured or self.model is None:
            print("Gemini API not configured")
            return "TIDAK_TERBACA"
            
        try:
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            pil_image = Image.fromarray(image_rgb)
            
            prompt = self.get_ocr_prompt(label)
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    print(f"{label} - Attempt {attempt+1}")
                    response = self.model.generate_content([prompt, pil_image])
                    
                    if not response.text:
                        print("No response")
                        continue
                        
                    text = response.text.strip().upper()
                    print(f"{label} response: '{text}'")
                    
                    if 'container' in label.lower():
                        patterns = [
                            r"([A-Z]{4}\d{7})",           
                            r"([A-Z]{4}\s*\d{7})",        
                            r"([A-Z]{3,4}\d{6,8})",       
                        ]
                    elif 'plate' in label.lower():
                        patterns = [
                            r"([A-Z0-9\s]{6,12})",        
                            r"([A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3})",  
                            r"(\b[A-Z0-9]{6,10}\b)",      
                        ]
                    else:  
                        patterns = [
                            r"(.{4,20})",                 
                            r"(\b[A-Z0-9\s\-]+\b)",       
                        ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, text)
                        if match:
                            result = match.group(1).strip()
                            print(f"{label} matched: {result}")
                            
                            if len(result) >= 4:  
                                return result
                    
                    if any(word in text for word in ["TIDAK", "TERBACA", "CANNOT", "UNABLE"]):
                        print(f"{label} explicit TIDAK_TERBACA")
                        return "TIDAK_TERBACA"
                        
                except Exception as e:
                    print(f"{label} attempt {attempt+1} failed: {e}")
                    if attempt == max_retries - 1:
                        return "ERROR"
                    time.sleep(1)
            
            return "TIDAK_TERBACA"
            
        except Exception as e:
            print(f"Gemini OCR error: {e}")
            return "ERROR"

gemini_config = GeminiConfig()