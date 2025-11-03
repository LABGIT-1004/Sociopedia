"""
Image moderation utility for checking inappropriate content in images
Includes nudity detection and hate speech OCR
"""
import os
import tempfile
import logging
from PIL import Image
import numpy as np

# Force TensorFlow to use GPU if available
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow dynamic memory allocation

logger = logging.getLogger(__name__)

# Initialize models as None - will be loaded on first use
_kmeans_model = None
_mobilenet_model = None
_easyocr_reader = None

# Hate/violence keywords list
HATE_KEYWORDS = [
    # Hate/violence
    'kill', 'hate', 'attack', 'destroy', 'violence', 'terror', 'bomb', 'shoot', 'fight', 
    'racist', 'abuse', 'threat', 'murder', 'blood', 'war', 'die', 'death', 'hurt', 
    'explode', 'gun', 'knife', 'assault', 'lynch', 'genocide', 'slur', 'curse', 'insult',
    'extremist', 'hate speech', 'hatecrime', 'terrorist', 'execute', 'massacre', 'riot',
    'hostile', 'enemy', 'danger', 'dangerous', 'harm', 'injure', 'injury', 'stab', 'rape',
    'molest', 'harass', 'bully', 'bullying', 'torture', 'tortured', 'torturing', 'abduct',
    'kidnap', 'kidnapping', 'shooting', 'behead', 'beheading', 'burn', 'burning', 'arson',
    'bombing', 'explosion', 'explosive', 'grenade', 'gunman', 'gunfire', 'gunshot',
    # Vulgar/offensive
    'fuck', 'shit', 'bitch', 'bastard', 'asshole', 'dick', 'piss', 'crap', 'slut', 
    'whore', 'cunt', 'motherfucker', 'fucker', 'bullshit', 'douche', 'prick', 'jackass',
    'retard', 'moron', 'idiot', 'stupid', 'suck', 'screwed', 'damn', 'hell', 'jerk',
    'loser', 'scum', 'trash', 'pig', 'dog', 'rat', 'snake', 'cock', 'pussy', 'hoe',
    'fag', 'gay', 'lesbian', 'porn', 'sex', 'nude', 'naked', 'orgy', 'fetish', 'pervert',
    'perverted', 'obscene', 'explicit', 'lewd', 'vulgar', 'offensive', 'abusive',
    'sexist', 'homophobic', 'transphobic', 'hateful', 'disgusting', 'gross', 'creep',
    'creepy', 'weirdo', 'freak', 'maniac', 'psycho', 'lunatic', 'crazy', 'insane'
]

HATE_KEYWORDS = [kw.lower() for kw in HATE_KEYWORDS]


def load_nudity_detection_model():
    """Load the nudity detection model (k-means clustering)"""
    global _kmeans_model, _mobilenet_model
    
    if _kmeans_model is None:
        try:
            import joblib
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
            
            # Configure GPU memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ GPU ENABLED - Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
            else:
                print("⚠️ No GPU found, using CPU")
            
            # Path to the model file
            model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'kmeans_model.joblib')
            
            if os.path.exists(model_path):
                _kmeans_model = joblib.load(model_path)
                _mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
                logger.info("Nudity detection model loaded successfully")
            else:
                logger.warning(f"Nudity detection model not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading nudity detection model: {str(e)}")
    
    return _kmeans_model, _mobilenet_model


def load_ocr_reader():
    """Load the EasyOCR reader"""
    global _easyocr_reader
    
    if _easyocr_reader is None:
        try:
            import easyocr
            import torch
            
            # Check if CUDA is available for PyTorch (used by EasyOCR)
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                print(f"✅ EasyOCR GPU ENABLED - CUDA available with {torch.cuda.device_count()} device(s)")
            else:
                print("⚠️ EasyOCR using CPU - CUDA not available")
            
            _easyocr_reader = easyocr.Reader(['en'], gpu=gpu_available)
            logger.info(f"EasyOCR reader loaded successfully (GPU: {gpu_available})")
        except Exception as e:
            logger.error(f"Error loading EasyOCR reader: {str(e)}")
    
    return _easyocr_reader


def extract_image_features(image_path):
    """Extract features from image using MobileNetV2"""
    try:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        
        _, model = load_nudity_detection_model()
        if model is None:
            return None
        
        img = load_img(image_path, target_size=(224, 224))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        feat = model.predict(arr, verbose=0)[0]
        return feat.reshape(1, -1)
    except Exception as e:
        logger.error(f"Error extracting image features: {str(e)}")
        return None


def check_nudity(image_path):
    """
    Check if image contains nudity using k-means clustering
    Returns: (is_adult, confidence, cluster_id)
    
    Model trained on 18,667 images (9,174 adult + 9,493 safe)
    - Clusters 0, 3: Adult content (98.1% and 89.8% adult respectively)
    - Clusters 1, 2, 4: Safe content (84.4%, 64.3%, and 97.9% safe respectively)
    """
    try:
        kmeans, _ = load_nudity_detection_model()
        if kmeans is None:
            logger.warning("Nudity detection model not available, skipping check")
            return False, 0.0, None
        
        # Extract features
        features = extract_image_features(image_path)
        if features is None:
            return False, 0.0, None
        
        # Predict cluster
        cluster = kmeans.predict(features)[0]
        
        # Cluster mapping based on training with 18,667 images (9,174 adult + 9,493 safe)
        # Cluster 0: 98.1% adult content
        # Cluster 3: 89.8% adult content
        # Clusters 1, 2, 4: Safe content (84.4%, 64.3%, 97.9% safe respectively)
        is_adult = cluster in [0, 3]
        
        # Calculate confidence (simplified - distance to cluster center)
        distances = kmeans.transform(features)[0]
        confidence = 1.0 - (distances[cluster] / np.max(distances)) if np.max(distances) > 0 else 0.0
        
        # Log to console for debugging
        print(f"🔍 NUDITY CHECK - Cluster: {cluster}, Confidence: {confidence:.2f}, Is Adult: {is_adult}, Blocking: {is_adult}")
        logger.info(f"Nudity check - Cluster: {cluster}, Confidence: {confidence:.2f}, Blocking: {is_adult}")
        
        return is_adult, confidence, int(cluster)
    except Exception as e:
        logger.error(f"Error checking nudity: {str(e)}")
        return False, 0.0, None


def extract_text_from_image(image_path):
    """Extract text from image using EasyOCR"""
    try:
        reader = load_ocr_reader()
        if reader is None:
            logger.warning("OCR reader not available, skipping text extraction")
            return ""
        
        result = reader.readtext(image_path, detail=0)
        text = '\n'.join(result)
        logger.info(f"Extracted text from image: {text[:100]}...")
        print(f"Extracted text from image: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return ""


def check_hate_speech_in_text(text):
    """
    Check if text contains hate speech or violent keywords
    Returns: (found_keywords, is_inappropriate)
    """
    import re
    
    if not text:
        return [], False
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in HATE_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            found_keywords.append(keyword)
    
    is_inappropriate = len(found_keywords) > 0
    
    if is_inappropriate:
        logger.warning(f"Hate speech keywords found: {found_keywords}")
    
    return found_keywords, is_inappropriate


def moderate_image(image_file):
    """
    Main function to moderate an image
    Checks for both nudity and hate speech in text
    
    Args:
        image_file: Django UploadedFile object or file path
    
    Returns:
        dict: {
            'is_inappropriate': bool,
            'reasons': list of str,
            'details': {
                'nudity': {
                    'detected': bool,
                    'confidence': float,
                    'cluster': int
                },
                'hate_speech': {
                    'detected': bool,
                    'keywords': list,
                    'extracted_text': str
                }
            }
        }
    """
    temp_path = None
    
    try:
        print(f"🔍 IMAGE MODERATION CALLED - Starting moderation check...")
        
        # Save uploaded file to temporary location
        if hasattr(image_file, 'temporary_file_path'):
            temp_path = image_file.temporary_file_path()
        else:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                for chunk in image_file.chunks():
                    tmp.write(chunk)
                temp_path = tmp.name
        
        print(f"🔍 Temporary image saved to: {temp_path}")
        
        reasons = []
        details = {
            'nudity': {'detected': False, 'confidence': 0.0, 'cluster': None},
            'hate_speech': {'detected': False, 'keywords': [], 'extracted_text': ''}
        }
        
        # Check for nudity
        print(f"🔍 Checking for nudity...")
        is_adult, confidence, cluster = check_nudity(temp_path)
        details['nudity'] = {
            'detected': is_adult,
            'confidence': float(confidence),
            'cluster': cluster
        }
        
        print(f"🔍 Nudity result: is_adult={is_adult}, cluster={cluster}, confidence={confidence:.2f}")
        
        # Block if image is classified as adult content (cluster 0 or 3)
        # Don't use confidence threshold - cluster classification is reliable enough
        if is_adult:
            reasons.append(f"Inappropriate visual content detected (cluster: {cluster}, confidence: {confidence:.2f})")
            print(f"❌ BLOCKING IMAGE - Adult content detected in Cluster {cluster}!")
        
        # Extract text and check for hate speech
        extracted_text = extract_text_from_image(temp_path)
        details['hate_speech']['extracted_text'] = extracted_text
        
        if extracted_text:
            keywords, is_hate = check_hate_speech_in_text(extracted_text)
            details['hate_speech']['detected'] = is_hate
            details['hate_speech']['keywords'] = keywords
            
            if is_hate:
                reasons.append(f"Inappropriate text detected in image: {', '.join(keywords[:3])}")
        
        return {
            'is_inappropriate': len(reasons) > 0,
            'reasons': reasons,
            'details': details
        }
        
    except Exception as e:
        logger.error(f"Error moderating image: {str(e)}")
        # In case of error, allow the image but log the error
        return {
            'is_inappropriate': False,
            'reasons': [],
            'details': {'error': str(e)}
        }
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path) and not hasattr(image_file, 'temporary_file_path'):
            try:
                os.unlink(temp_path)
            except:
                pass


def should_block_image(image_file):
    """
    Simple wrapper that returns True if image should be blocked
    """
    result = moderate_image(image_file)
    return result['is_inappropriate'], result['reasons'], result['details']
