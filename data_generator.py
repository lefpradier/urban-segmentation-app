import numpy as np
import keras
from tensorflow.python.keras.preprocessing import image

cats = {
    "void": [0, 1, 2, 3, 4, 5, 6],
    "flat": [7, 8, 9, 10],
    "construction": [11, 12, 13, 14, 15, 16],
    "object": [17, 18, 19, 20],
    "nature": [21, 22],
    "sky": [23],
    "human": [24, 25],
    "vehicle": [26, 27, 28, 29, 30, 31, 32, 33, -1],
}


image_dir = "aug_dataset/images"
mask_dir = "aug_dataset/masks"
image_list = os.listdir(image_dir)
mask_list = os.listdir(mask_dir)
image_list.sort()
mask_list.sort()
print(
    f". . . . .Number of images: {len(image_list)}\n. . . . .Number of masks: {len(mask_list)}"
)

# sanity check
for i in range(len(image_list)):
    assert image_list[i][16:] == mask_list[i][24:]


batch_size = 16
samples = 50000
steps = samples // batch_size
img_height, img_width = 256, 256
classes = 8
filters_n = 64


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        list_IDs,
        labels,
        batch_size=32,
        dim=(32, 32, 32),
        n_channels=1,
        n_classes=10,
        shuffle=True,
        aug=None,
    ):
        "Initialization"
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.aug = aug
        self.on_epoch_end()

    #!LEN
    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    #!GET ITEM
    # TODO: NEW
    def __getitem__(self, idx):
        idx = np.random.randint(0, 50000, batch_size)  # TODO : wtf 50000
        batch_x, batch_y = [], []
        drawn = 0
        for i in idx:
            #!chargement de l'image
            _image = cv2.imread(f"{image_dir}/{image_list[i]}")
            img = cv2.imread(f"{mask_dir}/{mask_list[i]}", cv2.IMREAD_COLOR)
            if self.aug is not None:
                augmented = self.aug(image=_image, mask=img)
                _image, img = augmented["image"], augmented["mask"]
            # _image = (
            #     image.img_to_array(
            #         image.load_img(
            #             f"{image_dir}/{image_list[i]}",
            #             target_size=(img_height, img_width),
            #         )
            #     )
            #     / 255.0  # TODO : redimen entre 0 et 1 et pas entre 0 et 255
            # )
            # ?TRANSFORMATION IMAGE

            #!RENDRE LISIBLE LE MASQUE
            #!chargement du masque
            # img = image.img_to_array(
            #     image.load_img(
            #         f"{mask_dir}/{mask_list[i]}",
            #         grayscale=True,
            #         target_size=(img_height, img_width),
            #     )
            # )
            # ?TRANSFORMATION MASK

            #!nombre de catégories observées sur le masque
            labels = np.unique(img)
            #!changer d'image si trop peu de catégories
            if len(labels) < 3:
                idx = np.random.randint(0, 50000, batch_size - drawn)
                continue
            #!transforme le masque pour chaque sous px de l'image en 8 cat avec  0
            img = np.squeeze(img)
            mask = np.zeros((img.shape[0], img.shape[1], 8))
            #!comparaison aux catégories théo def plus hauts cats
            # ajout 1 si présent
            #!pour les elements de l'image qui m'intresse
            for i in range(-1, 34):
                if i in cats["void"]:
                    mask[:, :, 0] = np.logical_or(mask[:, :, 0], (img == i))
                elif i in cats["flat"]:
                    mask[:, :, 1] = np.logical_or(mask[:, :, 1], (img == i))
                elif i in cats["construction"]:
                    mask[:, :, 2] = np.logical_or(mask[:, :, 2], (img == i))
                elif i in cats["object"]:
                    mask[:, :, 3] = np.logical_or(mask[:, :, 3], (img == i))
                elif i in cats["nature"]:
                    mask[:, :, 4] = np.logical_or(mask[:, :, 4], (img == i))
                elif i in cats["sky"]:
                    mask[:, :, 5] = np.logical_or(mask[:, :, 5], (img == i))
                elif i in cats["human"]:
                    mask[:, :, 6] = np.logical_or(mask[:, :, 6], (img == i))
                elif i in cats["vehicle"]:
                    mask[:, :, 7] = np.logical_or(mask[:, :, 7], (img == i))
                    #!reshape en 2D : input model?
            mask = np.resize(mask, (img_height * img_width, 8))
            #!append les images segmentées (mask) et les images brutes _image en couples pour input modèle
            batch_y.append(mask)
            batch_x.append(_image)
            drawn += 1
        return np.array(batch_x), np.array(batch_y)

    #!SHUFFLE
    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


#!transformations
light = A.Compose(
    [
        A.HorizontalFlip(p=1),
        A.RandomSizedCrop((800 - 100, 800 + 100), 600, 600),
        A.GaussNoise(var_limit=(100, 150), p=1),
    ],
    bbox_params=bbox_params,
    p=1,
)

medium = A.Compose(
    [
        A.HorizontalFlip(p=1),
        A.RandomSizedCrop((800 - 100, 800 + 100), 600, 600),
        A.MotionBlur(blur_limit=17, p=1),
    ],
    bbox_params=bbox_params,
    p=1,
)

strong = A.Compose(
    [
        A.HorizontalFlip(p=1),
        A.RandomSizedCrop((800 - 100, 800 + 100), 600, 600),
        A.RGBShift(p=1),
        A.Blur(blur_limit=11, p=1),
        A.RandomBrightness(p=1),
        A.CLAHE(p=1),
    ],
    bbox_params=bbox_params,
    p=1,
)

#!application des transfo
random.seed(13)
r = augment_and_show(
    light,
    image,
    labels,
    bboxes,
    instance_labels,
    titles,
    thickness=2,
    font_scale_orig=2,
    font_scale_aug=1,
)
