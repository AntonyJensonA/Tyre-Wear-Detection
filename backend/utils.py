import cv2
import numpy as np
from ultralytics import YOLO
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CBAM DEFINITION (Required for Model Loading) ---
class CBAMLayer(layers.Layer):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=8, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        
        # Channel Attention
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.mlp = models.Sequential([
            layers.Dense(channels // reduction, activation='relu', use_bias=False),
            layers.Dense(channels, use_bias=False)
        ])
        self.sigmoid_channel = layers.Activation('sigmoid')

        # Spatial Attention
        self.conv_spatial = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        channel_att = layers.Reshape((1, 1, self.channels))(channel_att)
        x = x * channel_att

        avg_out_s = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out_s = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_att = layers.Concatenate(axis=-1)([avg_out_s, max_out_s])
        spatial_att = self.conv_spatial(spatial_att)
        x = x * spatial_att
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels, "reduction": self.reduction})
        return config

def calculate_tire_wear_v2(image, mask):
    """
    Refined logic based on U-Net feature density + Edge complexity.
    This trusts the U-Net mask as the primary health indicator.
    """
    if len(image.shape) == 4: image = image[0] # remove batch dim
    if len(mask.shape) == 4: mask = mask[0]

    # 1. Mask Density Analysis
    mask_area = np.sum(mask > 0.5)
    total_area = mask.size
    if total_area == 0: return 100.0
    mask_density = mask_area / total_area
    
    # 2. Structural Edge Complexity
    gray_img = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_img, 50, 150)
    
    # Only count edges that fall WITHIN the U-Net mask
    mask_binary = (mask > 0.5).squeeze().astype(np.uint8)
    if mask_area == 0: return 100.0
    
    struct_edges = cv2.bitwise_and(edges, edges, mask=mask_binary)
    edge_density = np.sum(struct_edges > 0) / (mask_area + 1e-6)
    
    # 3. Balanced Health Calculation
    # Baseline: ~35% mask density, ~12% edge complexity
    if mask_density < 0.005: 
        return 100.0 # Truly zero features = 100% worn
        
    health_score = (mask_density / 0.35) * 70 + (edge_density / 0.12) * 30
    health_score = min(max(health_score, 0), 100)
    
    return float(100 - health_score)

# Load YOLOv8 model for filtering (Nano version for speed)
# We use a global variable to avoid reloading for every request
yolo_model = YOLO("yolov8n.pt") 

# Forbidden COCO classes that we want to reject
# 0: person, 14: bird, 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, etc.
FORBIDDEN_CLASSES = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

IMG_SIZE = 128

def letterbox_resize(img, size=(IMG_SIZE, IMG_SIZE)):
    """Resizes image to `size` preserving aspect ratio by adding black padding."""
    h, w = img.shape[:2]
    scale = min(size[0]/h, size[1]/w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    y_off = (size[0] - new_h) // 2
    x_off = (size[1] - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def detect_and_crop_tire(img):
    """OpenCV heuristic method to detect and crop the largest dark circular object (the tire)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    return img[y:y+h, x:x+w]

def preprocess_image(image_bytes):
    """Prepares image for U-Net input."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Using the verified logic from your notebook: resize whole image to maintain context
    resized = letterbox_resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    input_tensor = resized / 255.0
    return np.expand_dims(input_tensor, axis=0)

def extract_textured_tire(input_tensor, mask):
    """Applies the U-Net mask to the original RGB image for CNN classification."""
    # Ensure mask is binarized
    clean_mask = (mask > 0.5).astype(np.float32)
    # Multiply RGB by Mask
    return input_tensor * clean_mask

def is_tire_present(mask, threshold=0.01):
    """
    Checks if the predicted mask contains enough 'tire' pixels to be considered a tire.
    A threshold of 0.01 means at least 1% of the image must be covered by the mask.
    """
    # Normalize mask to binary 0 or 1
    binary_mask = (mask > 0.5).astype(np.float32)
    # Calculate the ratio of mask pixels to total pixels
    mask_ratio = float(np.sum(binary_mask) / (binary_mask.shape[1] * binary_mask.shape[2]))
    print(f"DEBUG: Mask ratio = {mask_ratio:.4f} (Threshold = {threshold})")
    return mask_ratio >= threshold

def detect_objects_yolo(image_bytes):
    """
    Uses YOLOv8 to detect forbidden objects (humans, animals, birds).
    Returns (is_filtered, detected_objects_list)
    """
    # Convert bytes to numpy image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return False, []

    # Run inference
    results = yolo_model(img, verbose=False)[0]
    
    detected_classes = results.boxes.cls.cpu().numpy().astype(int)
    is_filtered = False
    detected_names = []

    for cls_id in detected_classes:
        class_name = yolo_model.names[cls_id]
        detected_names.append(class_name)
        if cls_id in FORBIDDEN_CLASSES:
            is_filtered = True
            
    return is_filtered, detected_names
