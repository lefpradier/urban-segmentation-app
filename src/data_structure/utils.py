from itertools import combinations, chain
import os
import random
import shutil


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
