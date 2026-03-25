import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
#------
from google.colab import drive
drive.mount('/content/drive')
#------
data_path = '/content/drive/MyDrive/TyreNet'
#------
IMG_SIZE = 128
def load_data(base_path):
    images = []
    labels = []

    categories = ["Defective", "Good"]

    for category in categories:
        path = os.path.join(base_path, category)

        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)

            # 🔥 FINAL LABEL SYSTEM
            if category == "Good":
                labels.append(0.9)
            else:
                labels.append(0.3)

    images = np.array(images) / 255.0
    labels = np.array(labels)

    return images, labels


images, labels = load_data(data_path)

print("Images:", images.shape)
print("Labels:", labels.shape)

#------
def create_mask(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)

    return mask

masks = np.array([create_mask(img) for img in images])

print("Masks:", masks.shape)
#------
X_train, X_test, M_train, M_test, y_train, y_test = train_test_split(
    images, masks, labels, test_size=0.2, random_state=42
)
#------
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(input_shape)

    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)

    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss=lambda y_true, y_pred: dice_loss(y_true, y_pred) +
                                   tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )

    return model

unet = build_unet()
unet.summary()
#------
unet.fit(
    X_train, M_train,
    validation_data=(X_test, M_test),
    epochs=50,
    batch_size=4
)
#------
def get_clean_mask(model, images):
    preds = model.predict(images)
    return (preds > 0.5).astype(np.float32)

train_masks = get_clean_mask(unet, X_train)
test_masks = get_clean_mask(unet, X_test)
#------
def attention_block(x):
    attention = layers.Conv2D(1, 1, activation='sigmoid')(x)
    return layers.Multiply()([x, attention])

def build_cnn():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))

    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = attention_block(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
    )

    return model

cnn = build_cnn()
cnn.summary()
#------

#------
cnn.fit(
    train_masks, y_train,
    validation_data=(test_masks, y_test),
    epochs=30,
    batch_size=4
)
#------
def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    mask = unet.predict(img)
    mask = (mask > 0.5).astype(np.float32)

    pred = cnn.predict(mask)[0][0]
    pred = np.clip(pred, 0, 1)

    life_percent = pred * 100
    wear_percent = 100 - life_percent
    remaining_km = pred * 40000

    return wear_percent, life_percent, remaining_km
#------
wear, life, km = predict('/content/drive/MyDrive/dummytest/t1.jpg')

print(f"Wear: {wear:.2f}%")
print(f"Life: {life:.2f}%")
print(f"Remaining Distance: {km:.0f} km")
#------
# Save UNet
unet_model.save('/content/drive/MyDrive/unet_tyre_model.h5')

# Save CNN regression model
cnn_reg_model.save('/content/drive/MyDrive/cnn_regression_attention.h5')