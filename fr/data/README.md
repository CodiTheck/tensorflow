## Les données
![](https://img.shields.io/badge/lastest-2023--03--01-success)
![](https://img.shields.io/badge/status-en%20r%C3%A9daction%20-yellow)

Donc, si tu ne le sais pas encore, les données constituent une partie importante de l'apprentissage automatique ! En fait, c'est tellement important que la plupart des activités de ce cours porteront sur l'exploration, le nettoyage et la sélection des données appropriées.

<br/>
<details id="table-content" open>
    <summary>Table des Contenus</summary>
    <ul>
        <li><a href="#importation-des-modules">Importation des modules</a> </li>
        <li><a href="#travailler-avec-les-données">Travailler avec les données</a>
            <ul>
                <li><a href="#récupération-des-données">Récupération des données</a></li>
                <li><a href="#exploration-des-données">Exploration des données</a></li>
            </ul>
        </li>
    </ul>

</details>
<br/>

### Importation des modules
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

### Travailler avec les données
Le jeu de données (dataset) sur lequel nous allons travailler est celui du Titanic. Il contient des tonnes d'informations sur chaque passager du navire. Notre première étape sera d'explorer les données afin de les comprendre. C'est donc ce que nous allons faire ! <br/>

Il s'agit donc essentiellement de prédire qui va survivre, ou la probabilité que quelqu'un survive au Titanic, à partir d'un ensemble d'informations. On a donc besoin de charger cet ensemble de données.<br/>

Ci-dessous, nous allons charger le jeu de données et apprendre comment l'explorer à l'aide de certains outils intégrés.<br/>

#### Récupération des données
La fonction `pd.read_csv()` nous renvoie un nouvelle instance de `DataFrame` de pandas. Vous pouvez considérer un *dataframe* comme un tableau. C'est donc avec cette fonction qu'on charger notre ensemble de donnée encore appeler **dataset** en Anglais. <br/>

Avec la fonction `pop()`, on va extraire la colonne `"survived"` de notre dataset pour la stocker dans une nouvelle variable (`y_train` et `y_eval`). Cette colonne nous indique simplement si la personne a survécu ou non.<br/>

```python
# On va utiliser pandas  pour charger les données
# qui sont disponibles au format CSV (Comma Separated Values).
# On va charger les données pour l'entrainement du modèle:
df_train = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")

# On va charger les données pour l'évaluation du modèle:
df_eval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")

# Après le chargement, on affiche juste les données
# qu'on va utiliser pour l'entraînement.
print(df_train)

y_train = df_train.pop("survived")
y_eval = df_eval.pop("survived")

# Note:
# les raisons pour lesquelles la dataset a été séparée en deux
# est l'évaluation et le calcule de pourcentage de précision
# du modèle après entrainement.
```

Ce code produit le résultat ci-dessous:

```
     survived     sex   age  n_siblings_spouses  parch     fare   class     deck  embark_town alone
0           0    male  22.0                   1      0   7.2500   Third  unknown  Southampton     n
1           1  female  38.0                   1      0  71.2833   First        C    Cherbourg     n
2           1  female  26.0                   0      0   7.9250   Third  unknown  Southampton     y
3           1  female  35.0                   1      0  53.1000   First        C  Southampton     n
4           0    male  28.0                   0      0   8.4583   Third  unknown   Queenstown     y
..        ...     ...   ...                 ...    ...      ...     ...      ...          ...   ...
622         0    male  28.0                   0      0  10.5000  Second  unknown  Southampton     y
623         0    male  25.0                   0      0   7.0500   Third  unknown  Southampton     y
624         1  female  19.0                   0      0  30.0000   First        B  Southampton     y
625         0  female  28.0                   1      2  23.4500   Third  unknown  Southampton     n
626         0    male  32.0                   0      0   7.7500   Third  unknown   Queenstown     y

[627 rows x 10 columns]
```

Et voici à quoi ressemble notre ensemble de données ! Je sais que cela semble être un tas de charabia, 
mais c'est comme ça que nous devons le charger.<br/>

#### Exploration des données
Pour commencer, on va afficher les dimensions de notre dataset, c'est à dire, afficher les nombres de lignes et de colonnes.

```python
sh = df_train.shape
print(sh)  # Ce qui affiche: (627, 9)

```

Donc, on a 627 éléments ou lignes et 9 caractéristiques (variables) ou colonnes observées sur chacun de ces éléments. <br/>

Nous avons donc nos colonnes, qui représentent simplement les différents attributs ou variables dans notre ensemble de données.
Et de ces différents attributs de notre ensemble de données, nous avons la colonne `"survived"`. Ce sont les valeur de cette dernière
qu'on va essayer de prédire avec notre modèle. On va donc appeller cette colonne **notre étiquette**. Ainsi, ici, `0` signifie
que la personne n'a pas survécu, et `1` signifie que la personne a survécu.<br/> 

Concernant la colonne des informations sur les survivants qui avait été extraite,
elle est belle et bien stockée dans la variable `y_train`.

```python
print(y_train.head())

```

```
0    0
1    1
2    1
3    1
4    0
Name: survived, dtype: int64
```

Pour afficher les données, on peut utiliser la méthode `head()` de l'instance `DataFrame`. Elle permet tout simplement 
d'afficher les 5 premiers éléments en tête de liste de notre dataframe.

```python
head = df_train.head()
print(head)

```

```
      sex   age  n_siblings_spouses  parch     fare  class     deck  embark_town alone
0    male  22.0                   1      0   7.2500  Third  unknown  Southampton     n
1  female  38.0                   1      0  71.2833  First        C    Cherbourg     n
2  female  26.0                   0      0   7.9250  Third  unknown  Southampton     y
3  female  35.0                   1      0  53.1000  First        C  Southampton     n
4    male  28.0                   0      0   8.4583  Third  unknown   Queenstown     y
```

Si nous voulons une analyse statistique de nos données, nous pouvons utiliser la méthode `describe()`.

```python
desc = df_train.describe()
print(desc)

```

```
              age  n_siblings_spouses       parch        fare
count  627.000000          627.000000  627.000000  627.000000
mean    29.631308            0.545455    0.379585   34.385399
std     12.511818            1.151090    0.792999   54.597730
min      0.750000            0.000000    0.000000    0.000000
25%     23.000000            0.000000    0.000000    7.895800
50%     28.000000            0.000000    0.000000   15.045800
75%     35.000000            1.000000    0.000000   31.387500
max     80.000000            8.000000    5.000000  512.329200
```

Maintenant, en y réfléchissant seul pendant une seconde, et en regardant certaines des catégories que nous avons ici, peux tu penser à la raison pour laquelle la régression linéaire serait un bon algorithme pour quelque chose comme ça ? Eh bien, analysons un peut notre
ensemble de données.

- Si le passager est une femme, on peut supposer qu'elle aura plus de chances de survivre, juste parce que, tu le sais, la façon dont notre culture fonctionne, les femmes et les enfants sont sauvés d'abord, on est d'accord ? Et si tu regarde bien cet ensemble de données, tu remarquera que lorsqu'il s'agit d'une femme, il est assez rare qu'elle n'ait pas survécue. D'ailleurs, essayons d'afficher le graphe du pourcentage des survivants en fonction du sexe.


```python
# On reconstitue la dataset avec la colonne "survived"
# pour qu'on puisse compter.
ds_with_survived_column = [df_train, y_train]
pd.concat(ds_with_survived_column, axis=1).groupby('sex')\
        .survived.mean()\
        .plot(kind='barh')\
        .set_xlabel('% survive')

plt.show()

```

![](./images/Figure_6.png)

- Maintenant, si nous regardons l'âge.

```python
df_train.age.hist(bins=20)
plt.show()

```

![](./images/Figure_3.png)

Peut on penser comment l'âge pourrait avoir d'inffluance sur les résultats ? Eh bien, je suppose que si le passager est beaucoup plus jeune, alors il a probablement plus de chances de survivre, parce qu'il serait, comme tu le sais déjà, prioritaire pour être secouru par un canots de sauvetage ou quoi que ce soit, je ne sais pas grand-chose. Je ne peux donc tirer aucune conclusion à ce sujet.
J'essaie juste de passer en revue les attributs pour t'expliquer pourquoi nous devons choisir l'algorithme de la régression linéaire.

- Le nombre de frères et sœurs (`n_siblings_spouses`) n'influe pas forcément sur la prédiction, à mon avis.
- Donc la colone `"class"` porte les informnations sur la classe de chaque passager dans le bateau. Il y avait trois (3), première classe, 
deuxième classe et troisième classe.

```python
pd.concat(ds_with_survived_column, axis=1).groupby('class')\
        .survived.mean()\
        .plot(kind='barh')\
        .set_xlabel('% survive')

plt.show()

```

![](./images/Figure_7.png)

En exploitant le graphe ci-dessus, On pourrait donc penser qu'un passager qui fait partie d'une classe supérieure a plus de chances de survivre.

- Concernant la colonne `"alone"`, reprenons le code de représentation graphique ci-dessus et remplace tout simplement 
`'class'` par `'alone'` et exécute le code ensuite. Moi, lorsque j'exécute, j'ai le graphe ci-dessous:

![](./images/Figure_8.png)

Après analyse, on constate que c'est ceux qui ont voyagés seul qui ont eu plus de change.<br/>


```python
# Pour représenter les effectifs totaux au niveau des deux sexes.
df_train.sex.value_counts().plot(kind='barh')
plt.show()

```

![](./images/Figure_4.png)

```python
# Pour représenter les effectifs totaux au niveau des trois classes.
df_train['class'].value_counts().plot(kind='barh')
plt.show()

```

![](./images/Figure_5.png)


Après avoir analysé toutes ces informations, on note ce qui suit:
- La majorité des passagers ont entre 20 et 30 ans.
- La majorité des passagers sont des hommes.
- La majorité des passagers sont en "troisième" classe.
- La majorité des passagers qui ont survécues sont des femme.
- La majorité des passagers qui ont survécues on voyagé seul.
- La majorité des passagers qui ont survécues font partie de la première classe (classe supérieur).


<br/>
<br/>

- Je passe à la session **suivante**: [Deep Learning](../deep_learning/README.md)
- [<--](../generalities/README.md) Je reviens à la session **précédente**: [Installation et configuration](../installation/README.md)

