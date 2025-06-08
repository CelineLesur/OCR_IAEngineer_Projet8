import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
import random

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images_list, masks_list, batch_size=16, image_size=(224, 224),
                 augmentations=None, shuffle=True, encoder=None, **kwargs):
        super().__init__(**kwargs)
        self.images_list = images_list
        self.masks_list = masks_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.encoder = encoder
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.images_list) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        images = []
        masks = []
        # images_origin = []

        for i in batch_indexes:
            # === Chargement image et masque ===
            img_path = self.images_list[i]
            mask_path = self.masks_list[i]
            image_origin = cv2.imread(img_path)
            # image_origin_copy = image_origin.copy()
            if image_origin is None:
                raise ValueError(f"Image non trouvée : {img_path}")
            image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
            mask_origin = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


            # === Mapping des ID vers les 8 super-categories ===
            id_to_supercat = {
                0: 'vide', 1: 'vide', 2: 'vide', 3: 'vide', 4: 'vide', 5: 'vide', 6: 'vide',
                7: 'route/trottoir', 8: 'route/trottoir', 9: 'route/trottoir',10: 'route/trottoir',
                11: 'construction', 12: 'construction', 13: 'construction',14: 'construction',
                15: 'construction', 16: 'construction', 17: 'objet', 18: 'objet',19: 'objet',
                20: 'objet', 21: 'nature', 22: 'nature', 23: 'ciel', 24: 'humain',25: 'humain',
                26: 'vehicule', 27: 'vehicule', 28: 'vehicule',29: 'vehicule', 30: 'vehicule',
                31: 'vehicule', 32: 'vehicule', 33: 'vehicule'
            }

            supercat_to_id = {
                'vide': 0, 'route/trottoir': 1, 'construction': 2, 'objet': 3,
                'nature': 4, 'ciel': 5, 'humain': 6, 'vehicule': 7
            }

            # Appliquer le mapping à chaque pixel
            mapped_mask = np.vectorize(lambda x: supercat_to_id[id_to_supercat.get(x, 'vide')])(mask_origin)


            # === Resize ===
            image_origin = cv2.resize(image_origin, self.image_size)
            mask_origin = cv2.resize(mapped_mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            # === Prétraitement image ===
            img = image_origin.astype(np.float32)
            if self.encoder == 'vgg16':
                img = preprocess_input(img)
            else:
                img = img / 255.0

            # === Masque one-hot ===
            mask = tf.one_hot(mask_origin, depth=8)
            mask = tf.cast(mask, tf.float32)
            mask = tf.reshape(mask, (self.image_size[0], self.image_size[1], 8))

            images.append(img)
            masks.append(mask.numpy())
            # images_origin.append(image_origin_copy)

            # === Augmentation si demandé ===
            if self.augmentations:
                augmented = self.augmentations(image=image_origin, mask=mask_origin)
                image_aug = augmented['image']
                mask_aug = augmented['mask']

                # === Resize au cas où augmentation altère la taille ===
                image_aug = cv2.resize(image_aug, self.image_size)
                mask_aug = cv2.resize(mask_aug, self.image_size, interpolation=cv2.INTER_NEAREST)

                # === Normalisation image ===
                image_aug = image_aug.astype(np.float32)
                if self.encoder == 'vgg16':
                    image_aug = preprocess_input(image_aug)
                else:
                    image_aug = image_aug / 255.0

                # === One-hot pour le masque ===
                mask_aug = tf.one_hot(mask_aug, depth=8)
                mask_aug = tf.cast(mask_aug, tf.float32)
                mask_aug = tf.reshape(mask_aug, (self.image_size[0], self.image_size[1], 8))

                images.append(image_aug)
                masks.append(mask_aug.numpy())
                # images_origin.append(image_origin_copy)

        # return np.stack(images, axis=0), np.stack(masks, axis=0), np.stack(images_origin, axis=0)
        return np.stack(images, axis=0), np.stack(masks, axis=0)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def get_original_images_and_masks(self, index):
        """
        Récupère les images originales (non prétraitées, non augmentées)
        et leurs masques correspondants pour visualisation.
        """
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        original_images = []
        original_masks = []

        for i in batch_indexes:
            img_path = self.images_list[i]
            mask_path = self.masks_list[i]

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            # Remapping identique
            id_to_supercat = {
                0: 'vide', 1: 'vide', 2: 'vide', 3: 'vide', 4: 'vide', 5: 'vide', 6: 'vide',
                7: 'route/trottoir', 8: 'route/trottoir', 9: 'route/trottoir',10: 'route/trottoir',
                11: 'construction', 12: 'construction', 13: 'construction',14: 'construction',
                15: 'construction', 16: 'construction', 17: 'objet', 18: 'objet',19: 'objet',
                20: 'objet', 21: 'nature', 22: 'nature', 23: 'ciel', 24: 'humain',25: 'humain',
                26: 'vehicule', 27: 'vehicule', 28: 'vehicule',29: 'vehicule', 30: 'vehicule',
                31: 'vehicule', 32: 'vehicule', 33: 'vehicule'
            }

            supercat_to_id = {
                'vide': 0, 'route/trottoir': 1, 'construction': 2, 'objet': 3,
                'nature': 4, 'ciel': 5, 'humain': 6, 'vehicule': 7
            }

            mapped_mask = np.vectorize(lambda x: supercat_to_id[id_to_supercat.get(x, 'vide')])(mask)

            original_images.append(image)
            original_masks.append(mapped_mask)

        return original_images, original_masks

