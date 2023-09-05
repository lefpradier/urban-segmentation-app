import numpy as np
import tensorflow as tf
import cv2
import albumentations as A


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        img_list,
        mask_list,
        batch_size=32,
        shuffle=True,
        aug_list=None,
        img_height=256,
        img_width=256,
    ):
        "Initialization"
        #!batch size relative transfo ou pas
        if aug_list is not None:
            self.batch_size = batch_size / 2
        else:
            self.batch_size = batch_size
        self.img_list = img_list
        self.mask_list = mask_list
        self.shuffle = shuffle
        self.img_height = img_height
        self.img_width = img_width
        self.cats = {
            "void": [0, 1, 2, 3, 4, 5, 6],
            "flat": [7, 8, 9, 10],
            "construction": [11, 12, 13, 14, 15, 16],
            "object": [17, 18, 19, 20],
            "nature": [21, 22],
            "sky": [23],
            "human": [24, 25],
            "vehicle": [26, 27, 28, 29, 30, 31, 32, 33, -1],
        }

        #!FILTER DICT
        filter_dict = {
            "hflip": A.HorizontalFlip(p=1),
            "rgb": A.RGBShift(p=1),
            "rotate": A.ShiftScaleRotate(p=1),
            "blur": A.Blur(blur_limit=11, p=1),
            "bright": A.RandomBrightness(p=1),
            "contrast": A.CLAHE(p=1),
            "mblur": A.MotionBlur(blur_limit=17, p=1),
        }
        if aug_list is not None:
            self.aug = A.Compose([filter_dict[x] for x in aug_list])
        else:
            self.aug = None
        self.on_epoch_end()

    #!LEN
    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(len(self.img_list) / float(self.batch_size)))

    #!GET ITEM
    # TODO: NEW
    def __getitem__(self, idx):
        idx = [
            i
            for i in range(
                idx * self.batch_size,
                min((idx + 1) * self.batch_size, len(self.img_list)),
            )
        ]
        batch_x, batch_y = [], []
        drawn = 0
        for i in idx:
            #!chargement de l'image
            _image = cv2.imread(self.img_list[i])
            #!chargement du masque
            img = cv2.imread(self.mask_list[i], cv2.IMREAD_COLOR)
            #!resize img
            img = cv2.resize(img, (self.img_height, self.img_width))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _image = cv2.resize(_image, (self.img_height, self.img_width)) / 255.0
            #!pour appliquer la suite sur img et aug
            for j in range(2):
                if self.aug is not None and j > 0:
                    augmented = self.aug(image=_image, mask=img)
                    _image, img = augmented["image"], augmented["mask"]
                elif j > 0:
                    break
                #!RENDRE LISIBLE LE MASQUE
                #!transforme le masque pour chaque sous px de l'image en 8 cat avec  0
                img = np.squeeze(img)
                mask = np.zeros((img.shape[0], img.shape[1], 8))
                #!comparaison aux catégories théo def plus hauts cats
                # ajout 1 si présent
                #!pour les elements de l'image qui m'intresse
                for k in range(-1, 34):
                    if k in self.cats["void"]:
                        mask[:, :, 0] = np.logical_or(mask[:, :, 0], (img == k))
                    elif k in self.cats["flat"]:
                        mask[:, :, 1] = np.logical_or(mask[:, :, 1], (img == k))
                    elif k in self.cats["construction"]:
                        mask[:, :, 2] = np.logical_or(mask[:, :, 2], (img == k))
                    elif k in self.cats["object"]:
                        mask[:, :, 3] = np.logical_or(mask[:, :, 3], (img == k))
                    elif k in self.cats["nature"]:
                        mask[:, :, 4] = np.logical_or(mask[:, :, 4], (img == k))
                    elif k in self.cats["sky"]:
                        mask[:, :, 5] = np.logical_or(mask[:, :, 5], (img == k))
                    elif k in self.cats["human"]:
                        mask[:, :, 6] = np.logical_or(mask[:, :, 6], (img == k))
                    elif k in self.cats["vehicle"]:
                        mask[:, :, 7] = np.logical_or(mask[:, :, 7], (img == k))
                        #!reshape en 2D : input model?
                # mask = np.resize(mask, (self.img_height * self.img_width, 8))
                #!append les images segmentées (mask) et les images brutes _image en couples pour input modèle
                batch_y.append(mask)
                batch_x.append(_image)
                drawn += 1
        return np.array(batch_x), np.array(batch_y)

    #!SHUFFLE
    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.img_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)