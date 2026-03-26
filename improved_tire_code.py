import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import drive

# --- CELL 1: SETUP ---
# drive.mount('/content/drive')
# data_path = '/content/drive/MyDrive/TyreNet'
IMG_SIZE = 128

# --- CELL 2: CBAM LAYER ---
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
        # Channel Attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        channel_att = layers.Reshape((1, 1, self.channels))(channel_att)
        x = x * channel_att

        # Spatial Attention
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

# --- CELL 3: IMPROVED CBAM U-NET ---
def build_cbam_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return CBAMLayer(filters)(x)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    b = conv_block(p2, 128)

    # Decoder
    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c2])
    c3 = conv_block(u1, 64)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = conv_block(u2, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- CELL 4: WEAR FORMULA & FEATURE EXTRACTION ---
def calculate_tire_wear(image, mask):
    """
    Refined logic based on U-Net feature density + Edge complexity.
    This version trusts the U-Net mask (shown to be high quality) as the primary health indicator.
    """
    # 1. Mask Density Analysis
    # The U-Net segments high-health tread blocks. Density drop = wear.
    mask_area = np.sum(mask > 0.5)
    total_area = mask.size
    mask_density = mask_area / total_area
    
    # 2. Structural Edge Complexity
    # Isolate tread region and find sharp texture edges
    gray_img = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_img, 50, 150)
    # Only count edges that fall WITHIN the U-Net mask
    mask_binary = (mask > 0.5).squeeze().astype(np.uint8)
    struct_edges = cv2.bitwise_and(edges, edges, mask=mask_binary)
    edge_density = np.sum(struct_edges > 0) / (mask_area + 1e-6)
    
    # 3. Balanced Health Calculation
    # Healthy baseline: ~35% mask density, ~12% edge complexity within mask
    health_score = (mask_density / 0.35) * 70 + (edge_density / 0.12) * 30
    health_score = min(max(health_score, 0), 100)
    
    wear_percentage = 100 - health_score
    
    # For visualization, we keep the masks
    grooves = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    grooves = cv2.bitwise_and(grooves, grooves, mask=mask_binary)
    
    return float(wear_percentage), grooves, struct_edges

import os

# --- CELL 5: DATA PREPARATION & ENHANCED MASK GENERATION ---
def create_mask(img):
    """
    Generates a valid and clean tread mask using pre-processing and adaptive thresholding.
    1. Bilateral filtering for edge-preserving noise reduction.
    2. Adaptive Gaussian thresholding for lighting robustness.
    3. Morphological operations for artifact removal.
    """
    # Convert RGB image (0-1 float) to uint8 Gray
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # 1. Bilateral Filter: Removes noise but keeps edges sharp (vital for tread)
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Adaptive Thresholding: Handles shadows and dusty surface variations
    mask = cv2.adaptiveThreshold(
        gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 3. Morphological Operations: Opening to remove noise, Closing to fill gaps
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return np.expand_dims(mask / 255.0, axis=-1)

def load_and_preprocess_data(data_path):
    """
    Walks through Good and Defective folders, loads images, and generates masks.
    """
    images = []
    masks = []
    
    categories = ["Good", "Defective"]
    print(f"Starting data loading from: {data_path}")
    
    for category in categories:
        cat_path = os.path.join(data_path, category)
        if not os.path.exists(cat_path):
            print(f"Warning: Category folder {category} not found at {cat_path}")
            continue
            
        print(f"Processing category: {category}...")
        for filename in sorted(os.listdir(cat_path)):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(cat_path, filename)
                img = cv2.imread(img_path)
                if img is None: continue
                
                # Preprocess image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                
                # Generate valid mask automatically
                mask = create_mask(img)
                
                images.append(img)
                masks.append(mask)
                
    print(f"Data loading complete. Total images: {len(images)}")
    return np.array(images), np.array(masks)

# --- CELL 6: TRAINING ---
# model = build_cbam_unet()
# print("Loading and generating masks...")
# X, M = load_and_preprocess_data('/content/drive/MyDrive/TyreNet')
# if len(X) > 0:
#     print("Starting training...")
#     model.fit(X, M, epochs=50, batch_size=8, validation_split=0.2, 
#               callbacks=[tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)])
# else:
#     print("No data found. Please check your Drive path.")

# --- CELL 7: INFERENCE & VISUALIZATION ---
def visualize_improved_output(model, test_image):
    img_input = np.expand_dims(test_image, axis=0)
    pred_mask = model.predict(img_input)[0]
    
    wear, groove_map, edge_map = calculate_tire_wear(test_image, pred_mask)
    health = 100 - wear

    if health > 75: status = "Good Condition"
    elif health > 45: status = "Moderate Wear"
    elif health > 20: status = "Worn Out"
    else: status = "Critical/Flat"
    
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 4, 1)
    plt.title("Input Tyre")
    plt.imshow(test_image)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("CBAM-U-Net Segmentation")
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Structural Texture Map")
    plt.imshow(groove_map, cmap='magma')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title(f"ANALYSIS RESULT\nStatus: {status}\nWear: {wear:.1f}% | Health: {health:.1f}%")
    plt.imshow(edge_map, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
# --- CELL 8: SAVE MODEL FOR BACKEND ---
# model.save('/content/drive/MyDrive/cbam_unet_tire_v4.keras')
print("Model export ready. Use the above command to save to your Drive.")
