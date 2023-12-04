import numpy as np
import tensorflow as tf
import cv2
import albumentations as A
import random
import matplotlib.pyplot as plt

plt.style.use("custom_dark")


def px_count(image):
    """
    Counts pixels in an image
    ---
    Parameters:
    - image: np.array
    """
    cond = np.where(
        (image[:, :, 0] != 0) | (image[:, :, 1] != 0) | (image[:, :, 2] != 0), 1, 0
    )
    npx = np.sum(cond)
    return npx


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras on the fly with possibility for multithreading"

    def __init__(
        self,
        img_list,
        mask_list,
        batch_size=32,
        shuffle=True,
        aug_list=None,
        img_height=256,
        img_width=256,
        mosaic=False,
        oversampling=False,
        seed=42,
        clim=0.2,
        blim=0.2,
        oversampling_n=0,
        oversampling_max=1000,
        attention_mask=False,
        attention_mask_size=False,
    ):
        """
        Initialization
        ---
        Parameters:
        - img_list: list of image paths
        - mask_list: list of mask paths
        - batch_size: int
        - shuffle: boolean
        - aug_list: list of strings to choose among the keys of the filter_dict dictionary
        - img_height: int
        - img_width: int
        - mosaic: boolean, choose whether mosaic augmentation should be applied
        - oversampling: boolean, choose whether oversampling of minority classes should be applied
        - seed: int
        - clim: float between 0 and 1, range of contrast augmentation
        - blim: float between 0 and 1, range of brightness augmentation
        - oversampling_n: int, number of possible oversampled instances in an image
        - oversampling_max: int, area (number of pixels) under which oversampling can be applied
        - attention_mask: boolean, choose whether attention mask augmentation should be applied
        - attentin_mask_size: boolean, choose whether attention mask augmentation based on size should be applied
        """
        #!batch size TOUJOURS LA MÊME CAR SOIT AUG SOIT IMG
        self.batch_size = batch_size
        self.img_list = img_list
        self.mask_list = mask_list
        self.shuffle = shuffle
        self.img_height = img_height
        self.img_width = img_width
        self.attention_mask = attention_mask
        self.attention_mask_size = attention_mask_size
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
        self.mosaic = mosaic
        self.oversampling = oversampling
        self.oversampling_n = oversampling_n
        if oversampling_n > 0:
            self.oversampling = True
        self.oversampling_max = oversampling_max
        self.nepoch = -1
        random.seed(seed)
        np.random.seed(2 * seed)
        #! FILTER DICT: Dictionary among which image augmentation methods can be chosen
        filter_dict = {
            "hflip": A.HorizontalFlip(p=0.5),
            "rgb": A.RGBShift(p=0.5),
            "rotate": A.ShiftScaleRotate(p=0.5),
            "blur": A.Blur(blur_limit=11, p=0.5),
            "contrast": A.CLAHE(p=0.5),
            "mblur": A.MotionBlur(blur_limit=7, p=0.5),
            "rotateb": A.Rotate(limit=30, p=0.5),
            "rdcrop": A.RandomCrop(
                height=768, width=1536, p=0.5
            ),  #! 3/4 of original size
            "upcrop": A.Crop(x_min=0, x_max=2048, y_min=0, y_max=512, p=0.5),
            "rdtile": A.RandomCrop(
                height=256, width=512, p=0.5
            ),  #! 1/4 of original size
            "gnoise": A.GaussNoise(p=0.5),
            "bricon": A.RandomBrightnessContrast(
                brightness_limit=blim, contrast_limit=clim, p=0.5
            ),
        }
        if aug_list is not None:
            self.aug = A.Compose(
                [filter_dict[x] for x in aug_list if x in filter_dict.keys()]
            )
        else:
            self.aug = None
        self.on_epoch_end()

    #!LEN
    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(len(self.img_list) / float(self.batch_size)))

    #!SHUFFLE
    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.img_list))
        self.nepoch += 1
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    #####################################################MOSAIC AUGMENTATION
    #!GET ITEM
    def __getitem__(self, idx):
        "Selects and preprocesses items that will be sent to Keras on a given batch"
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
            #! Load image
            _image = cv2.imread(self.img_list[self.indexes[i]])
            #! Load mask in gray levels
            img = cv2.imread(self.mask_list[self.indexes[i]], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #! Check if the image contains underrepresented classes (useful for some augmentation methods)
            contains_small = False
            if np.any(
                np.isin(img, self.cats["human"]) | np.isin(img, self.cats["object"])
            ):
                contains_small = True
            #! Apply augmentation techniques
            if self.aug is not None:
                # Apply augmentation with Albumentations
                augmented = self.aug(image=_image, mask=img)
                # Upcrop technique: make sure the image contains underrepresented classes
                if augmented["image"].shape[0] / augmented["image"].shape[1] < 0.5:
                    augmenter = A.Compose([A.RandomCrop(height=512, width=1024, p=1)])
                    augmented = augmenter(
                        image=augmented["image"], mask=augmented["mask"]
                    )
                if (
                    "RandomCrop" in str(self.aug)
                    and "height=256" in str(self.aug)
                    and contains_small
                ):
                    trials = 0
                    while trials < 20:
                        if np.any(
                            np.isin(img, self.cats["human"])
                            | np.isin(img, self.cats["object"])
                        ):
                            break
                        augmented = self.aug(image=_image, mask=img)
                        trials += 1
                _image, img = augmented["image"], augmented["mask"]
            #! Resize img
            img = cv2.resize(img, (self.img_width, self.img_height))
            _image = cv2.resize(_image, (self.img_width, self.img_height))
            img = np.squeeze(img)
            #! Append the masks and raw images to the input pile for Keras
            batch_y.append(img)
            batch_x.append(_image / 255.0)
            drawn += 1
        # *𝗢𝗩𝗘𝗥𝗦𝗔𝗠𝗣𝗟𝗜𝗡𝗚 #################################################
        if self.oversampling:
            #! Isolate the pixels corresponding to the underrepresented classes
            coord = {"human": [], "object": []}
            indice = 0
            for _image, img in zip(batch_x, batch_y):
                #! Get the pixels for each of the underrepresented classes
                for c in ["human", "object"]:
                    if np.any(
                        np.isin(img, self.cats[c])
                    ):  # imgs with pixels in the concerned class
                        sampled_mask = np.where(
                            np.isin(img, self.cats[c]), img, 0
                        )  # Fill the other classes with zeroes
                        #!work on 3D of each images (RGB > mask only got 2 because of BW encoding)
                        sampled_img = np.zeros((_image.shape[0], _image.shape[1], 3))
                        sampled_img[:, :, 0] = np.where(
                            np.isin(img, self.cats[c]), _image[:, :, 0], 0
                        )
                        sampled_img[:, :, 1] = np.where(
                            np.isin(img, self.cats[c]), _image[:, :, 1], 0
                        )
                        sampled_img[:, :, 2] = np.where(
                            np.isin(img, self.cats[c]), _image[:, :, 2], 0
                        )
                        coord[c].append((sampled_mask, sampled_img, indice))
                indice += 1
            #! Copy-pasting of the isolated pixels on the original images
            for c in coord.keys():
                for patch_y, patch_x, index in coord[c]:
                    for rep in range(self.oversampling_n):
                        npx = px_count(patch_x)
                        if npx > 0:
                            scale_min = self.oversampling_max / npx
                            scale_max = 100 / npx
                            #! Transformation of the isolated pixels: rotation, enlargement, mirror
                            trials = 0
                            while trials < 20:
                                trials += 1
                                aug_mask = A.Compose(
                                    [
                                        A.Affine(
                                            scale=(scale_max, scale_min),
                                            translate_percent={
                                                "x": (0, 1),
                                                "y": (0, 1),
                                            },
                                            p=1,
                                            keep_ratio=True,
                                        )
                                    ]
                                )
                                temp = aug_mask(image=patch_x, mask=patch_y)
                                npx = px_count(temp["image"])
                                tempup = temp["image"][
                                    : int(2 * self.img_height / 3), :, :
                                ]
                                npxup = px_count(tempup)
                                # Count pixels where the mask isn't empty and contains overrepresented categories
                                cond = np.where(
                                    (
                                        (temp["image"][:, :, 0] != 0)
                                        | (temp["image"][:, :, 1] != 0)
                                        | (temp["image"][:, :, 2] != 0)
                                    )
                                    & (
                                        np.isin(
                                            batch_y[index],
                                            self.cats["flat"]
                                            + self.cats["void"]
                                            + self.cats["sky"]
                                            + self.cats["construction"],
                                        )
                                    ),
                                    1,
                                    0,
                                )
                                npx_overlap = np.sum(cond)
                                #! Count proportion of pixels in the upper part of the image
                                if npx_overlap > 0.67 * npx:
                                    xbis, ybis = temp["image"], temp["mask"]
                                    #! Paste the isolated pixels
                                    batch_x[index][:, :, 0] = np.where(
                                        ybis == 0,
                                        batch_x[index][:, :, 0],
                                        xbis[:, :, 0],
                                    )
                                    batch_x[index][:, :, 1] = np.where(
                                        ybis == 0,
                                        batch_x[index][:, :, 1],
                                        xbis[:, :, 1],
                                    )
                                    batch_x[index][:, :, 2] = np.where(
                                        ybis == 0,
                                        batch_x[index][:, :, 2],
                                        xbis[:, :, 2],
                                    )
                                    #!MASK have 2D > batch_y doesn't have to be explicit
                                    batch_y[index] = np.where(
                                        ybis == 0, batch_y[index], ybis
                                    )
                                    break
                                    # if trials too large, then abandon
        # ⁡⁢⁢⁣*𝗠𝗢𝗦𝗔𝗜𝗖 #################################################⁡
        if self.mosaic:
            xb = []
            yb = []
            for idx in range(self.batch_size):
                #! Choose 4 images
                if len(batch_x) >= 4:
                    indices = random.sample([idx for idx in range(len(batch_x))], k=4)
                else:
                    indices = random.choices([idx for idx in range(len(batch_x))], k=4)
                #! Paste them together on a 4x4 template
                # creation of the template
                patch_x = np.zeros((2 * self.img_height, 2 * self.img_width, 3))
                patch_y = np.zeros((2 * self.img_height, 2 * self.img_width))
                # Fill the template
                # 1 img
                patch_x[self.img_height :, : self.img_width, :] = batch_x[indices[0]]
                patch_y[self.img_height :, : self.img_width] = batch_y[indices[0]]
                # 2
                patch_x[self.img_height :, self.img_width :, :] = batch_x[indices[1]]
                patch_y[self.img_height :, self.img_width :] = batch_y[indices[1]]
                # 3
                patch_x[: self.img_height, : self.img_width, :] = batch_x[indices[2]]
                patch_y[: self.img_height, : self.img_width] = batch_y[indices[2]]
                # 4
                patch_x[: self.img_height, self.img_width :, :] = batch_x[indices[3]]
                patch_y[: self.img_height, self.img_width :] = batch_y[indices[3]]
                #! Cut a window in the template with albumentation
                crop = A.Compose(
                    [A.RandomCrop(height=self.img_height, width=self.img_width, p=1)]
                )
                # Get augmented mask
                temp = crop(image=patch_x, mask=patch_y)
                xbis, ybis = temp["image"], temp["mask"]
                xb.append(xbis)
                yb.append(ybis)
            batch_x = xb
            batch_y = yb
        # *ATTENTION MASK ##########################################
        if self.attention_mask:
            for idx in range(len(batch_y)):
                #! Check if the image contains any of the underrepresented classes
                if np.any(
                    np.isin(batch_y[idx], self.cats["human"])
                    | np.isin(batch_y[idx], self.cats["object"])
                ):
                    #! Probability of selective attention decreases with epochs
                    p = 0.8
                    p = p - (0.8 / 30) * (self.nepoch)
                    # Apply the transformation probability to the images
                    if random.uniform(0, 1) < p:
                        #! Replace all the other pixels in black in the image (respectively 'flat' in the mask)
                        batch_x[idx][:, :, 0] = np.where(
                            np.isin(
                                batch_y[idx], self.cats["human"] + self.cats["object"]
                            ),
                            batch_x[idx][:, :, 0],
                            0,
                        )
                        batch_x[idx][:, :, 1] = np.where(
                            np.isin(
                                batch_y[idx], self.cats["human"] + self.cats["object"]
                            ),
                            batch_x[idx][:, :, 1],
                            0,
                        )
                        batch_x[idx][:, :, 2] = np.where(
                            np.isin(
                                batch_y[idx], self.cats["human"] + self.cats["object"]
                            ),
                            batch_x[idx][:, :, 2],
                            0,
                        )
                        batch_y[idx] = np.where(
                            np.isin(
                                batch_y[idx], self.cats["human"] + self.cats["object"]
                            ),
                            batch_y[idx],
                            7,
                        )  # flat
        if self.attention_mask_size:
            #! Define quantiles under which a class is allowed to be fed to the model
            cats_dist = {
                "void": [0.0649, 0.109, 0.124, 1],
                "flat": [0.360, 0.40, 0.433, 1],
                "construction": [0.137, 0.216, 0.295, 1],
                "object": [0.0076, 0.0149, 0.0256, 1],
                "nature": [0.053, 0.1306, 0.2128, 1],
                "sky": [0.00877, 0.027, 0.05, 1],
                "human": [0.000455, 0.0035, 0.012, 1],
                "vehicle": [0.0206, 0.056, 0.113, 1],
            }
            #! Different quantiles should be considered depending on the epoch
            size_idx = min(3, self.nepoch // 5)
            #! Initialize a list of indices to remove, so that no empty image is fed to the model
            idx_to_remove = []
            for idx in range(len(batch_y)):
                for c in self.cats:
                    freq = np.sum(np.isin(batch_y[idx], self.cats[c])) / (
                        batch_y[idx].shape[0] * batch_y[idx].shape[1]
                    )
                    if freq > cats_dist[c][size_idx]:
                        #! Replace all the unallowed pixels by black in the image (respectively 'flat' in the mask)
                        batch_x[idx][:, :, 0] = np.where(
                            np.isin(batch_y[idx], self.cats[c]),
                            batch_x[idx][:, :, 0],
                            0,
                        )
                        batch_x[idx][:, :, 1] = np.where(
                            np.isin(batch_y[idx], self.cats[c]),
                            batch_x[idx][:, :, 1],
                            0,
                        )
                        batch_x[idx][:, :, 2] = np.where(
                            np.isin(batch_y[idx], self.cats[c]),
                            batch_x[idx][:, :, 2],
                            0,
                        )
                        batch_y[idx] = np.where(
                            np.isin(batch_y[idx], self.cats[c]), batch_y[idx], 7
                        )  # flat
                # If the image contains only one class, it shouldn't be fed to the model
                if (
                    np.sum(np.isin(batch_y[idx], self.cats["flat"]))
                    == batch_y[idx].shape[0] * batch_y[idx].shape[1]
                ):
                    idx_to_remove.append(idx)
            # Remove black images from the batch input list
            batch_x = [
                batch_x[idx] for idx in range(len(batch_x)) if idx not in idx_to_remove
            ]
            batch_y = [
                batch_y[idx] for idx in range(len(batch_y)) if idx not in idx_to_remove
            ]
        # *################################################
        #! Simplify the mask: from 32 categories to 8 subcategories
        for idx in range(len(batch_y)):
            mask = np.zeros((batch_y[idx].shape[0], batch_y[idx].shape[1], 8))
            #!pour les elements de l'image qui m'intresse
            for k in range(-1, 34):
                if k in self.cats["void"]:
                    mask[:, :, 0] = np.logical_or(mask[:, :, 0], (batch_y[idx] == k))
                elif k in self.cats["flat"]:
                    mask[:, :, 1] = np.logical_or(mask[:, :, 1], (batch_y[idx] == k))
                elif k in self.cats["construction"]:
                    mask[:, :, 2] = np.logical_or(mask[:, :, 2], (batch_y[idx] == k))
                elif k in self.cats["object"]:
                    mask[:, :, 3] = np.logical_or(mask[:, :, 3], (batch_y[idx] == k))
                elif k in self.cats["nature"]:
                    mask[:, :, 4] = np.logical_or(mask[:, :, 4], (batch_y[idx] == k))
                elif k in self.cats["sky"]:
                    mask[:, :, 5] = np.logical_or(mask[:, :, 5], (batch_y[idx] == k))
                elif k in self.cats["human"]:
                    mask[:, :, 6] = np.logical_or(mask[:, :, 6], (batch_y[idx] == k))
                elif k in self.cats["vehicle"]:
                    mask[:, :, 7] = np.logical_or(mask[:, :, 7], (batch_y[idx] == k))
            batch_y[idx] = mask
        return np.array(batch_x), np.array(batch_y)
