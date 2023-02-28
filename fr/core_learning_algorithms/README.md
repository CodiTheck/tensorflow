![](https://img.shields.io/badge/lastest-2023--02--28-success)

## Algorithmes d'apprentissage fondamentaux
Dans cette note, nous allons parcourir 4 algorithmes fondamentaux d'apprentissage automatique. Nous appliquerons chacun de ces algorithmes à des problèmes et des ensembles de données uniques avant de mettre en évidence les cas d'utilisation de chacun.<br/>

Les algorithmes sur lesquels nous allons nous concentrer sont les suivants:
- Régression linéaire;
- Classification;
- Clusturing;
- Hidden Markov Models;

Il existe de nombreux outils au sein de TensorFlow qui pourraient être utilisés pour résoudre les problèmes que nous allons voir ci-dessous. J'ai choisi les outils qui, selon moi, offrent le plus de variété et sont les plus faciles à utiliser.

### Configuration et importation des modules
Avant de continuer, tu dois installer `scikit-learn` et importer les modules ci-dessous.

```sh
# ~$
pip install -q sklearn
```

> Dans ce cour et tous les autres cours à venir, si le symbole `~$` se trouve au début d'un block
> de code, cela signifie qu'il s'agit de lignes de commande qu'on peut exécuter dans un terminal.

La ligne suivante n'est pas requise sauf si tu travail dans un `notebook`.

```python
%tensorflow_version 2.x
```

```python
# Importation des modules.
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

# tensorflow
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

```

### Les données
Donc, si tu ne le sais pas encore, les données constituent une partie importante de l'apprentissage automatique !

### Régression linéaire
La régression linéaire est l'une des formes les plus fondamentales de l'apprentissage automatique et est utilisée pour prédire des valeurs numériques. Dans ce cours, nous allons utiliser un modèle linéaire pour prédire le taux de survie des passagers à partir de l'ensemble de données du Titanic.

#### Comment ça marche ?
Avant de nous plonger dans le vif du sujet, je vais te donner une explication très superficielle de l'algorithme de régression linéaire.<br/>

La régression linéaire suit un concept très simple. Si les points de données sont liés de façon linéaire, alors nous pouvons générer une ligne (droite) de meilleur ajustement pour ces points et l'utiliser pour prédire les futures valeurs.<br/>

Prenons l'exemple d'un ensemble de données (dataset) avec une caractéristique (variable) et une étiquette (sortie). C'est à dire $y = f(x)$.

```python
import numpy as np
import matplotlib.pyplot as plt

# On génère un ensemble de points sous la forme y = f(x).
X = [1, 2, 2.5, 3, 4]
Y = [1, 4, 7, 9, 15]

# On affiche une représentation graphique:
plt.plot(X, Y, 'ro')
plt.axis([0, 6, 0, 20])
plt.show()

```

![](./images/Figure_1.png)

Nous pouvons voir que ces données ont une tendance linéaire. Lorsque la valeur `x` augmente, la valeur `y` augmente également. Grâce à cette variation, nous pouvons établir une **ligne (droite)** de meilleur ajustement pour cet ensemble de données. Dans cet exemple, notre droite n'utilisera qu'une seule variable d'entrée, car nous travaillons dans le plan. Dans un ensemble de données plus grand avec plus de caractéristiques, notre droite aura plus de variables.

>"La droite de meilleur ajustement fait référence à une droite qui traverse un nuage de points de données et exprime une meilleur relation entre ces points."
> [Vas sur ce lien.](https://www.investopedia.com/terms/l/line-of-best-fit.asp)

Voici un rappel de l'équation d'une droite dans le plan.

$$ y = ax + b $$

Voici un exemple de droite de meilleur ajustement pour le graphe ci-dessus.

```python
plt.plot(X, Y, 'ro')
plt.axis([0, 6, 0, 20])

un_x = np.unique(X)
poly1d = np.poly1d(np.polyfit(X, Y, 1))

plt.plot(un_x, poly1d(un_x))
plt.show()

```

![](./images/Figure_2.png)

Une fois que nous avons généré cette ligne pour notre ensemble de données, nous pouvons utiliser son équation pour prédire les futures valeurs.



<br/>
<br/>

[<--](../README.md) Je reviens à la session **précédente**: [TensorFlow](../README.md)

