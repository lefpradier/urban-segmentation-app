from itertools import combinations, chain
import os
import random
import shutil
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D


def holdout(image_dir, mask_dir):
    random.seed(42)
    #!1.Comptage des éléments
    city_list = os.listdir(image_dir + "/train")
    city_count = []
    for i in city_list:
        temp = len(os.listdir(os.path.join(image_dir + "/train", i)))
        city_count.append(temp)
    # FOR TEST ~ 1/6 TRAIN beg
    min_thr = 1 / 10 * sum(city_count)
    max_thr = 1 / 6 * sum(city_count)
    #!2.Sélection des combinaisons de villes qui respectent diverses cdt
    # Build dict
    citytocount = {key: value for key, value in zip(city_list, city_count)}
    result = []
    # Parse combinations with more than 2 cities
    # 3 cities in VALID
    allCombinations = chain(
        *(combinations(city_list, i) for i in range(3, len(city_list) + 1))
    )
    for c in allCombinations:
        # Get count for this combination
        countForThisCombination = sum((citytocount[name] for name in c))
        # Test for min/max
        if countForThisCombination > min_thr and countForThisCombination < max_thr:
            result += [c]
    #!3. Sélection aléatoire de combinaisons parmis les précédentes
    city_test = random.choice(result)
    #!4. Create new test directory
    os.mkdir(image_dir + "/new_test/")
    os.mkdir(mask_dir + "/new_test/")
    #!5. redirect selected train files to test file
    for ville in city_test:
        for dir in [image_dir, mask_dir]:
            source = dir + "/train/" + str(ville)
            destination = dir + "/new_test/" + str(ville)
            shutil.move(source, destination)
    return city_test


def mini_unet(image_shape, num_of_classes):
    # ​‌‍‍from : https://github.com/YBIGTA/DL_Models/blob/master/models/unet/Keras%20(on%20Google%20Colaboratory)/mini%20u-net.ipynb​
    # Contracting Path
    input_image = Input(image_shape)
    conv1_1 = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv1_1",
    )(input_image)
    conv1_2 = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv1_2",
    )(conv1_1)
    pool_1 = MaxPooling2D(name="pool_1")(conv1_2)
    conv2_1 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2_1",
    )(pool_1)
    conv2_2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2_2",
    )(conv2_1)
    pool_2 = MaxPooling2D(name="pool_2")(conv2_2)
    conv3_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv3_1",
    )(pool_2)
    conv3_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv3_2",
    )(conv3_1)
    pool_3 = MaxPooling2D(name="pool_3")(conv3_2)
    conv4_1 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv4_1",
    )(pool_3)
    conv4_2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv4_2",
    )(conv4_1)

    # Expanding Path
    upconv5_1 = UpSampling2D(name="upconv5_1")(conv4_2)
    upconv5_2 = Conv2D(
        filters=128,
        kernel_size=(2, 2),
        activation="relu",
        padding="same",
        name="upconv5_2",
    )(upconv5_1)
    concat_5 = concatenate([upconv5_2, conv3_2], axis=3, name="concat_5")
    conv5_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv5_1",
    )(concat_5)
    conv5_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv5_2",
    )(conv5_1)
    upconv6_1 = UpSampling2D(name="upconv6_1")(conv5_2)
    upconv6_2 = Conv2D(
        filters=64,
        kernel_size=(2, 2),
        activation="relu",
        padding="same",
        name="upconv6_2",
    )(upconv6_1)
    concat_6 = concatenate([upconv6_2, conv2_2], axis=3, name="concat_6")
    conv6_1 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv6_1",
    )(concat_6)
    conv6_2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv6_2",
    )(conv6_1)
    upconv7_1 = UpSampling2D(name="upconv7_1")(conv6_2)
    upconv7_2 = Conv2D(
        filters=32,
        kernel_size=(2, 2),
        activation="relu",
        padding="same",
        name="upconv7_2",
    )(upconv7_1)
    concat_7 = concatenate([upconv7_2, conv1_2], axis=3, name="concat_7")
    conv7_1 = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv7_1",
    )(concat_7)
    conv7_2 = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv7_2",
    )(conv7_1)
    conv8 = Conv2D(
        filters=num_of_classes, kernel_size=(1, 1), activation="softmax", name="conv8"
    )(conv7_2)
    model = Model(inputs=input_image, outputs=conv8, name="model")
    return model
