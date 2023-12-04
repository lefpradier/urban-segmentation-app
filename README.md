# Urban segmentation app
Ce repository contient le code source permettant d'entraîner et de déployer un modèle de segmentation d'images urbaines.
Ce modèle est déployé grâce à une API FastAPI (exposée initialement à l'adresse suivante : https://urban-segmentation-api.azurewebsites.net/), et utilisé par une application Flask (initialement exposée à l'adresse suivante : https://urban-segmentation-app.azurewebsites.net/dashboard).
## Spécifications techniques
Le modèle a été entraîné et testé sur le jeu de données CityScapes (https://www.cityscapes-dataset.com/), avec des images au format 128x256 et 8 catégories principales. Il consiste d'une architecture FPN (<i>feature pyramid network</i>) avec un backbone EfficientNetB4.