from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os, random, shutil, matplotlib.pyplot as plt

# ── Parámetros de división y training ─────────────────────
BASE_DIR    = "classification/fairface_database/fairface_latino"
CLASSES     = ["Male", "Female"]
RANDOM_SEED = 42
BATCH       = 32
train_ratio = 0.8
val_ratio   = 0.1
test_ratio  = 0.1

# División de datos en train, valid y test
random.seed(RANDOM_SEED)

# ---------------------------
# 2) Crear carpetas destino
# ---------------------------
for subset in ("train", "valid", "test"):
    for cls in CLASSES:
        os.makedirs(os.path.join(BASE_DIR, subset, cls), exist_ok=True)

# ---------------------------
# 3) Dividir y copiar
# ---------------------------
for cls in CLASSES:
    src_folder = os.path.join(BASE_DIR, cls)
    all_files  = [f for f in os.listdir(src_folder)
                  if os.path.isfile(os.path.join(src_folder, f))]
    random.shuffle(all_files)

    n = len(all_files)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio   * n)

    train_files = all_files[:n_train]
    val_files   = all_files[n_train:n_train + n_val]
    test_files  = all_files[n_train + n_val:]

    for fname in train_files:
        shutil.copy2(
            os.path.join(src_folder, fname),
            os.path.join(BASE_DIR, "train", cls, fname)
        )
    for fname in val_files:
        shutil.copy2(
            os.path.join(src_folder, fname),
            os.path.join(BASE_DIR, "valid", cls, fname)
        )
    for fname in test_files:
        shutil.copy2(
            os.path.join(src_folder, fname),
            os.path.join(BASE_DIR, "test", cls, fname)
        )

print("✓ División completa: train/valid/test creadas bajo fairface_latino/")

# Rutas a los conjuntos de datos
train_dir = os.path.join(BASE_DIR, "train")
val_dir   = os.path.join(BASE_DIR, "valid")
test_dir  = os.path.join(BASE_DIR, "test")

# ── Generadores con augmentación robusta ──────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.7,1.4],
    shear_range=0.1,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "train"),
    target_size=(224,224),
    batch_size=BATCH,
    class_mode='categorical',
    seed=RANDOM_SEED
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "valid"),
    target_size=(224,224),
    batch_size=BATCH,
    class_mode='categorical',
    seed=RANDOM_SEED
)
test_gen = val_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "test"),
    target_size=(224,224),
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)

# ── Construcción del modelo con Dropout & BatchNorm ──────
base = MobileNetV2(include_top=False, input_shape=(224,224,3), weights='imagenet')
x    = GlobalAveragePooling2D()(base.output)
x    = BatchNormalization()(x)
x    = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)
model  = Model(base.input, output)

# ── Fase 1: entrenar solo las cabezas ─────────────────────
for layer in base.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint("best_gen_model.h5", save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks
)

# ── Fase 2: fine-tune de las últimas capas del base ───────
# Descongelar últimas N capas de la base para afinarlas
N = 30
for layer in base.layers[:-N]:
    layer.trainable = False
for layer in base.layers[-N:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(1e-5),   # LR más bajo
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)

# ── Evaluación final sobre test ───────────────────────────
model.load_weights("best_gen_model.h5")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"Test Accuracy: {test_acc:.3f}, Test Loss: {test_loss:.3f}")

# ── Graficar curvas combinadas ────────────────────────────
acc       = history.history['accuracy']      + history_fine.history['accuracy']
val_acc   = history.history['val_accuracy']  + history_fine.history['val_accuracy']
loss      = history.history['loss']          + history_fine.history['loss']
val_loss  = history.history['val_loss']      + history_fine.history['val_loss']
epochs    = range(1, len(acc)+1)

plt.figure()
plt.plot(epochs, acc,      label='Train Acc')
plt.plot(epochs, val_acc,  label='Val Acc')
plt.hlines(test_acc, 1, epochs[-1], label='Test Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss,     label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.hlines(test_loss, 1, epochs[-1], label='Test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ── Guardar el mejor modelo final ──────────────────────────
model.save("classification/clasificador_genero_latino_robust.h5")
