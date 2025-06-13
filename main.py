from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from azure.storage.blob import BlobServiceClient
from contextlib import asynccontextmanager
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, SGD
import segmentation_models as sm
import tensorflow as tf
import uvicorn
import tempfile

# Paramètres de connexion blob storage
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=stockaccountp8;AccountKey=flae3B4NIMDm7xc1N3pmP84VgN+zqnM0+HsGw/Y+OqhfomqVLftO9jy4J5r2aIn+eccsB1G8A147+AStRvQ6TA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "projet8"
BLOB_NAME = "unet_vgg16_aug_epoch50_model.weights.h5"
LOCAL_WEIGHTS_PATH = os.path.join("APImodel", "APImodel.weights.h5")

id_to_color = {
    0: (0, 0, 0),         # void - noir
    1: (128, 64, 128),    # flat - violet
    2: (70, 70, 70),      # construction - gris
    3: (255, 0, 0),       # object - rouge
    4: (0, 128, 0),       # nature - vert
    5: (70, 130, 180),    # sky - bleu ciel
    6: (220, 20, 60),     # human - rose
    7: (0, 0, 142),       # vehicle - bleu foncé
    }

def download_blob():
    print("Téléchargement des poids depuis Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    blob_client = container_client.get_blob_client(BLOB_NAME)

    with open(LOCAL_WEIGHTS_PATH, "wb") as f:
        blob_data = blob_client.download_blob()
        f.write(blob_data.readall())
    
    # with open(LOCAL_WEIGHTS_PATH, "wb") as f:
    #     data = blob_client.download_blob()
    #     data.readinto(f)
    print(">> Taille du fichier de poids téléchargé :", os.path.getsize(LOCAL_WEIGHTS_PATH), "octets")
    print("Poids téléchargés et sauvegardés localement.")

def decode_mask(mask_2d):
    # Transforme un masque 2D en image RGB selon les couleurs de classe
    h, w = mask_2d.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in id_to_color.items():
        mask_rgb[mask_2d == class_id] = color
    return mask_rgb

def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge et prétraite une image à partir de bytes pour le modèle.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    resized_img = cv2.resize(image_np, (224, 224))
    preprocessed_img = preprocess_input(resized_img)
    return resized_img, np.expand_dims(preprocessed_img, axis=0)
    
def build_unet(backbone=None, input_shape=(224, 224, 3), num_classes=8, filters=[32, 64],
               dropout_rate=0.0, optimizer='adam', learning_rate=1e-3, loss='dice'):

    vgg = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    vgg.trainable = False

    inputs = vgg.input
    skips = [
        vgg.get_layer("block1_conv2").output,
        vgg.get_layer("block2_conv2").output,
        vgg.get_layer("block3_conv3").output,
        vgg.get_layer("block4_conv3").output
    ]
    x = vgg.get_layer("block5_conv3").output

    for i, skip in reversed(list(enumerate(skips))):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(512 // (2**i), 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(512 // (2**i), 3, activation='relu', padding='same')(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
    base_model = models.Model(inputs, outputs)

    loss_fn = sm.losses.categorical_focal_dice_loss
    opt = Adam(learning_rate=learning_rate)

    base_model.compile(optimizer=opt, loss=loss_fn,
                       metrics=[sm.metrics.IOUScore(threshold=0.5), 'accuracy'])

    return base_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    download_blob()
    global model
    model = build_unet(
        backbone='vgg16',
        input_shape=(224, 224, 3),
        num_classes=8,
        filters=[64, 128, 256],
        dropout_rate=0.3,
        optimizer='adam',
        learning_rate=1e-4,
        loss='focal_dice'
    )

    # Charge les poids dans le modèle
    model.load_weights(LOCAL_WEIGHTS_PATH)
    app.state.model = model

    print("Modèle chargé au démarrage")
    yield
    print("Arrêt de l'application")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "API U-Net prête"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    original_img, input_tensor = preprocess_image(await image.read())

    # Prédiction
    prediction = app.state.model.predict(input_tensor)
    pred_mask = np.argmax(prediction[0], axis=-1)  # (224, 224)

    # Masque coloré
    color_mask = decode_mask(pred_mask)

    # Superposition
    overlay = (0.3 * original_img + 0.7 * color_mask).clip(0, 255).astype(np.uint8)

    # Création de la figure avec légende
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.axis("off")

    # Légende des couleurs
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(color) / 255.0)
               for color in id_to_color.values()]
    labels = [str(cls_id) for cls_id in id_to_color.keys()]
    ax.legend(handles, labels, title="Classes", loc="lower left")

    # Sauvegarder et retourner
    result_path = "prediction_overlay.png"
    plt.savefig(result_path, bbox_inches="tight")
    plt.close()

    return FileResponse(result_path, media_type="image/png")



# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
