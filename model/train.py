"""
Train MobileNetV2 on the Brain Tumor MRI dataset.
Run on Kaggle/Colab (GPU). Outputs:
  - brain_tumor_mobilenetv2.keras
  - class_names.json
Copy both files into backend/ after training.
"""
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD, EPOCHS_FINE = 10, 10
MODEL_PATH = "brain_tumor_mobilenetv2.keras"

DATA_DIR = "/kaggle/input/brain-tumor-mri-dataset"
if not os.path.exists(DATA_DIR):
    import kagglehub
    DATA_DIR = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
TRAIN_DIR, TEST_DIR = os.path.join(DATA_DIR, "Training"), os.path.join(DATA_DIR, "Testing")

kw = dict(image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, validation_split=0.2, subset="training", seed=SEED, **kw)
val_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, validation_split=0.2, subset="validation", seed=SEED, **kw)
test_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, shuffle=False, **kw)

class_names = train_ds.class_names
print("Classes:", class_names)
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

train_ds, val_ds, test_ds = [d.prefetch(tf.data.AUTOTUNE) for d in (train_ds, val_ds, test_ds)]

aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"), layers.RandomRotation(0.08),
    layers.RandomZoom(0.1), layers.RandomContrast(0.1)], name="augmentation")

base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

# NOTE: scaling to [-1, 1] is INSIDE the model, so the backend feeds raw 0-255 pixels.
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = aug(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
       callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
       callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)]

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=cbs)

base.trainable = True
for l in base.layers[:-30]:
    l.trainable = False
for l in base.layers:
    if isinstance(l, layers.BatchNormalization):
        l.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD + EPOCHS_FINE,
          initial_epoch=len(h1.history["loss"]), callbacks=cbs)

model = tf.keras.models.load_model(MODEL_PATH)
print("Test accuracy:", model.evaluate(test_ds)[1])
y_true = np.concatenate([np.argmax(y, 1) for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), 1)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
print(confusion_matrix(y_true, y_pred))
