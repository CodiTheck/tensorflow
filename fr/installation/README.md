## Installation et configuration
![](https://img.shields.io/badge/lastest-2023--03--03-success)
<!-- ![](https://img.shields.io/badge/status-en%20r%C3%A9daction%20-yellow)-->

On aura besoin d'installater les outils de la liste ci-dessous:
- [python3](https://www.python.org/downloads/): L'interpretteur permettant d'exécuter du script Python.
- [python3-pip](https://www.google.com/search?q=python3-pip): Le scripte permettant d'installer les modules python. Je pense que ce scripte n'est utilisable que sous Linux. Les autres systèmes d'exploitation n'aurons pas bsoin de `python3-pip` mais d'un autre package permettant d'avoir `pip` pour installer des modules sur ton système. Tu peux utiliser [anaconda](https://anaconda.org/anaconda/python) comme alternative. Moi personnellement, je n'utilise pas `anaconda`.
- [python3-tk](https://www.google.com/search?q=python3-tk): Le programme est un utilitaire pour linux qui permet d'utiliser le module `tkinter` pour afficher les interfaces graphiques sous python. Il faut l'installer afin de pouvoir utiliser matplotlib pour visualiser des graphes.
- [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html): Le sripte de gestion d'environnement virtuel pour python.
- [pandas](https://pandas.pydata.org/): La bibliothèque permettant de charger et manipuler des données massives.
- [numpy](https://numpy.org/): La bibliothèque numérique de calcul matricielle.
- [matplotlib](https://matplotlib.org/stable/index.html): La bibliothèque de base pour visualiser des graphismes (Courbe, diagramme en baton, histogramme, etc).
- [scikit-learn](https://scikit-learn.org/stable/): La bibliothèque d'apprentissage automatique (Machine Learning) et de pré-traitement de données la plus populaire pour Python. Elle contient tous les algorithmes de machine learning les plus utilisés, tout, sauf les algorithmes de Deep Learning.
- [tensorflow](https://www.tensorflow.org/install?hl=fr): La bibliothèque de deep learning. C'est elle qui fait l'objet de ce cour.

<br/>
<details id="table-content" open>
    <summary>Table des Contenus</summary>
    <ul>
        <li><a href="#sous-linux">Sous Linux</a>
            <ul>
                <li><a href="#installation">Installation</a>
                    <ul>
                        <li><a href="#installation-de-python">Installation de python</a></li>
                        <li><a href="#installation-de-virtualenv">Installation de virtualenv</a></li>
                        <li><a href="#installation-des-modules-fondamentaux">Installation des modules fondamentaux</a>
                            <ul>
                                <li><a href="#pandas">Pandas</a></li>
                                <li><a href="#numpy">Numpy</a></li>
                                <li><a href="#matplotlib">Matplotlib</a></li>
                                <li><a href="#scikit-learn">Scikit-learn</a></li>
                                <li><a href="#tensorflow">TensorFlow</a></li>
                            </ul>
                        </li>
                    </ul>
                <li>
            </ul>
        </li>
        <li><a href="#sous-windows-et-mac">Sous Windows et Mac</a></li>
    </ul>

</details>
<br/>

### Sous Linux
> Dans ce cour et tous les autres cours à venir, si le symbole `~$` se trouve au début d'un block de code, cela signifie qu'il s'agit de lignes de commande qu'on peut exécuter dans un terminal.

#### Installation
##### Installation de python

```sh
# ~$
# Il faut copier et coller simplement le contenu de toute 
# cette case dans ton terminal et appuier sur la touche 
# [ENTER] de ton clavier.
sudo apt install python3;\
sudo apt install python3-pip python3-tk
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
Pour installer `pandas`, exécute la commande suivant. Clique [ici](https://pandas.pydata.org/) si tu veux en savoir plus.

```sh
# ~$
pip install pandas
```

###### Numpy
Pour installer `numpy`, exécute la commande suivant. Clique [ici](https://numpy.org/) si tu veux en savoir plus.

```sh
# ~$
pip install numpy
```

###### Matplotlib
Pour installer `matplotlib`, exécute la commande suivant. Clique [ici](https://matplotlib.org/stable/index.html) si tu veux en savoir plus.

```sh
# ~$
pip install matplotlib
```

###### Scikit-learn
Pour installer `scikit-learn`, exécute la commande suivant. Clique [ici](https://scikit-learn.org/stable/) si tu veux en savoir plus.

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

### Sous Windows et Mac
Je suis un peu désolé de te décevoir. Personnellement, je ne suis pas un grand fan des logiciels propriétaire. Donc, je n'ai pas l'habitude de travailler sur Windows et Mac. Pour cela, je te recommande d'aller suis des tutoriels sur comment installer les outils que je t'ai listé en haut.

<br/>
<br/>


- Je passe à la session **suivante**: [Algorithmes d'apprentissage fondamentaux](../core_learning_algorithms/README.md)
- [<--](../generalities/README.md) Je reviens à la session **précédente**: [Généralités](../generalities/README.md) -->

<!--
<div style="width: 100%; display: flex; justify-content: center;  column-gap: 20px;">
    <div style="border: 1px solid; padding: 1em;"><a href="../generalities/README.md"><< TensorFlow</a></div>
    <div style="border: 1px solid; padding: 1em;"><a href="../core_learning_algorithms/README.md">Algorithmes d'apprentissage fondamentaux >></a></div>

</div>
-->
