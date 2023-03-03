## Installation et configuration
![](https://img.shields.io/badge/lastest-2023--03--03-success)
![](https://img.shields.io/badge/status-en%20r%C3%A9daction%20-yellow)

### Sous Linux
#### Installation
##### Installation de python

```sh
# ~$
# Il faut copier et coller simplement le contenu de toute 
# cette case dans ton terminal et appuier sur la touche 
# [ENTER] de ton clavier.
sudo apt install python3;\
sudo apt install python3-pip
```

Il faut s'assurer de la version de python qui est installée. La version de python
utilisée est `python 3.9.12`. Tu peux aussi utiliser la version `3.8`.

##### Installation de virtualenv
Il est fortement recommandé de travailler dans un environnement virtuel. Donc utilise une des commandes
suivant pour installer le gestionnaire d'environnement virtuel que je te propose.

```sh
# ~$
sudo apt install python3-venv
```

OU

```sh
# ~$
sudo pip3 install virtualenv
```

##### Installation des modules fondamentaux
Avant d'installer quoi que ce soit, il faut activer l'environnement virtuel. En fonction du choix 
d'installation du gestionnaire que tu a fait ci-dessus, exécute une des commandes suivante.

```sh
# ~$
# Si tu as installé le gestionnaire avec sudo apt install python3-venv,
# alors utilise ceci:
python3 -m venv env
```

OU

```sh
# ~$
# Si tu as installé le gestionnaire avec sudo pip3 install virtualenv,
# alors utilise ceci:
virtualenv env -p python3
```

###### Pandas
Pour installer `pandas`, exécute la commande suivant. Clique [ici]() si tu veux en savoir plus.

```sh
# ~$
pip install pandas
```

###### Matplotlib
Pour installer `matplotlib`, exécute la commande suivant. Clique [ici]() si tu veux en savoir plus.

```sh
# ~$
pip install matplotlib
```

###### Scikit-learn
Pour installer `scikit-learn`, exécute la commande suivant. Clique [ici]() si tu veux en savoir plus.

```sh
# ~$
pip install -q sklearn
```

###### TensorFlow
Si tu utilise **Ubuntu 20.04**, installe d'abord les ressources Python supplémentaires en exécutant les lignes de commande suivantes.

```sh
# ~$
# Il faut copier et coller simplement le contenu de toute 
# cette case dans ton terminal et appuier sur la touche 
# [ENTER] de ton clavier.
sudo apt -y update;\
sudo apt -y install python3 python3-pip python3-setuptools python3-dev python3-testresources
```

Pour ce qui concerne l'installation, je ne peux te fournir aucun tutoriel qui marche à tout les coups,
donc pour faire les choses simplement, je t'invite à consulter la documentation officielle qui se
trouve [par ici](https://www.tensorflow.org/install?hl=fr)

<br/>
<br/>

<!--
- Je passe à la session **suivante**: [Algorithmes d'apprentissage fondamentaux](../core_learning_algorithms/README.md)
- [<--](../generalities/README.md) Je reviens à la session **précédente**: [TensorFlow](../generalities/README.md) -->

<button>OK</button>
<div style="width: 100%; display: flex; justify-content: center;  column-gap: 20px;">
    <div style="border: 1px solid; padding: 1em;"><a href="../generalities/README.md"><< TensorFlow</a></div>
    <div style="border: 1px solid; padding: 1em;"><a href="../core_learning_algorithms/README.md">Algorithmes d'apprentissage fondamentaux >></a></div>

</div>

