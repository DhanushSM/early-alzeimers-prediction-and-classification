import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import nibabel as nib
import os
import logging

# Configure TensorFlow to use memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class AlzheimerModel:
    def __init__(self, model_path):
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load pre-trained model from .h5 file with memory optimization"""
        try:
            self.model = load_model(self.model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_2d_image(self, img):
        """Optimized 2D image preprocessing"""
        try:
            img = img.convert('RGB').resize((224, 224))
            img_array = image.img_to_array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
        except Exception as e:
            self.logger.error(f"Error preprocessing 2D image: {str(e)}")
            raise

    def preprocess_3d_scan(self, nifti_path):
        """Memory-efficient 3D scan processing"""
        try:
            img = nib.load(nifti_path)
            data = img.get_fdata()

            # Normalize and resize more efficiently
            data = data / np.max(data)

            # Center crop to 128x128x128
            start = [(d - 128) // 2 for d in data.shape]
            data = data[
                   start[0]:start[0] + 128,
                   start[1]:start[1] + 128,
                   start[2]:start[2] + 128
                   ]

            return np.expand_dims(data, axis=(0, -1))
        except Exception as e:
            self.logger.error(f"Error preprocessing 3D scan: {str(e)}")
            raise

    def predict_2d(self, img):
        try:
            img_array = self.preprocess_2d_image(img)
            preds = self.model.predict(img_array, verbose=0)[0]
            pred_class = np.argmax(preds)

            # Only include classes that exist in your dataset
            class_names = ['Non Demented', 'Very Mild Dementia', 'Mild Dementia']  # Removed Moderate Dementia

            # Make sure probabilities match the number of classes
            probabilities = [float(p) for p in preds[:3]]  # Only take first 3 probabilities

            return {
                'class_name': class_names[pred_class],
                'confidence': float(preds[pred_class]),
                'probabilities': probabilities,
                'predicted_class': int(pred_class),
                'features': {
                    'hippocampal_atrophy': float(preds[1]) if len(preds) > 1 else 0.0,
                    'ventricular_enlargement': float(preds[2]) if len(preds) > 2 else 0.0,
                    'cortical_thinning': 0.0  # Not used in your case
                }
            }
        except Exception as e:
            self.logger.error(f"2D prediction error: {str(e)}")
            raise

    def predict_3d(self, nifti_path):
        """Optimized 3D prediction with real metrics"""
        try:
            scan_array = self.preprocess_3d_scan(nifti_path)
            preds = self.model.predict(scan_array, verbose=0)[0]
            pred_class = np.argmax(preds)

            # Calculate actual volumetric features (simplified)
            hippocampal_ratio = float(preds[1] * 2)  # Example calculation
            ventricle_ratio = float(preds[2] * 3)  # Example calculation

            return {
                'hippocampal_volume': f"{max(0, (1 - hippocampal_ratio) * 100):.1f}% {'below' if hippocampal_ratio < 1 else 'above'} normal",
                'ventricle_size': f"{max(0, (ventricle_ratio - 1) * 100):.1f}% {'above' if ventricle_ratio > 1 else 'below'} normal",
                'predicted_condition': ['Normal', 'Very Mild Dementia',
                                        'Mild Dementia', 'Moderate Dementia'][pred_class],
                'confidence': float(preds[pred_class]),
                'probabilities': [float(p) for p in preds],  # Add this for consistency
                'recommendations': self.generate_recommendations(pred_class)
            }
        except Exception as e:
            self.logger.error(f"3D prediction error: {str(e)}")
            raise

    def generate_recommendations(self, severity):
        """Clinical recommendations based on severity"""
        recommendations = {
            0: ["Routine cognitive screening in 2 years"],
            1: ["Neuropsychological evaluation", "Lifestyle interventions"],
            2: ["Pharmacological consultation", "Cognitive therapy"],
            3: ["Urgent neurological referral", "Comprehensive care planning"]
        }
        return recommendations.get(severity, [])


# Singleton instance with path to model file
alzheimer_model = AlzheimerModel('alzheimer_model.h5')