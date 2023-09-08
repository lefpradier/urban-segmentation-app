import numpy as np
import random


#!1.isoler les images avec des classes sous ech
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
#!2.isoler les px concernés sur chacune de ces images=mask
# ?Chargement img et mask
#!chargement de l'image
for _image, img in zip(batch_x, batch_y):
    #!Recup des px concernés par la classe
    coord = {"human": [], "vehicule": [], "object": []}
    for c in ["human", "vehicule", "object"]:
        if np.any(np.isin(img, cats[c])):  # img avec des px dans ces cats
            sampled_mask = np.where(
                np.isin(img, cats[c]), img, 0
            )  # rempli les non cats par des 0
            sampled_img = np.where(np.isin(img, cats[c]), _image, 0)
            coord[c].append((sampled_mask, sampled_img))
#!3.Tirage aléatoire de x élements qui vont recevoir le mask (concerné et non concerné)
# lecture des masks des catégories
for c in coord.keys:
    for y, x in coord[c]:
        indices = random.choices([idx for idx in range(len(batch_x))], 1)
        for index in indices:
            #!4.Transformation du mask : augmentation (rotate, enlarge, miror)
            aug_mask = A.Compose(
                [
                    A.Affine(
                        scale=(0.5, 2),
                        translate_percent=(0, 0.5),
                        rotate=(-45, 45),
                        p=1,
                    )
                ]
            )
            # recup mask aug
            xbis, ybis = aug_mask(image=x, mask=y)
            #!5.Collage du mask en définissant le point d'ancrage
            # coller mask dans img tirée
            batch_x[index] = np.where(ybis == 0, batch_x[index], xbis)
            batch_y[index] = np.where(ybis == 0, batch_y[index], ybis)
