import cv2
import numpy as np

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
