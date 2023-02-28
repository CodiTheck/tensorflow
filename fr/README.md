# TensorFlow
![](https://img.shields.io/badge/version-2.9.1-orange)
![](https://img.shields.io/badge/lastest-2023--02--28-success)
![](https://img.shields.io/badge/contact-dr.mokira%40gmail.com-red)

TensorFlow est une bibliothèque **open source** de Machine Learning, créée par Google, permettant de développer et d’exécuter des applications de Machine Learning et de Deep Learning.<br/>

Créé par l’équipe **Google Brain** en *2011*, sous la forme d’un système propriétaire dédié au réseaux de neurones de Deep Learning, TensorFlow s’appelait à l’origine **DistBelief**. Par la suite, le code source de DistBelief a été modifié et cet outil est devenu une bibliothèque basée application. En **2015**, précisement le *9 novembre*, il a été renommé en **TensorFlow** et Google l’a rendu open source. Depuis lors, il a subi plus de 21000 modifications par la communication et est passé en version 1.0 en février 2017.

<br/>

## Les tenseurs
Un tenseur est une généralisation des vecteurs et des matrices à des dimensions potentiellement plus élevées. En interne, `TensorFlow` représente les tenseurs comme des tableaux à **n-dimensions** contenant des données de type de base. <br/>

Ne sois pas surpris que les tenseurs soient un aspect fondamental de TensorFlow. Ce sont les principaux objets qui sont passés et sont manipulés tout au long du programme. Chaque tenseur représente un calcul partiellement défini qui produira éventuellement une valeur. Les programmes TensorFlow fonctionnent en construisant un graphe d'objets `Tensor` qui détaille la façon dont les tenseurs sont liés.
L'exécution de différentes parties du graphe permet de générer des résultats.<br/>

Chaque tenseur a un type de données et une dimension (`shape`).
- Les types données peuvent être: `float32`, `int32`, `string` et autres types.
- `shape` représente les dimentions des données contenues dans le tenseur.

### Création d'un tenseur
Vous trouverez ci-dessous un exemple de création de différents tenseurs.

```python
import tensorflow as tf  # N'oublit pas d'importer la librairie.


string = tf.Variable("This is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

```

### Rang/degré d'un tenseur
Un autre mot pour le rang est le degré, ces termes signifient simplement le nombre de dimensions impliquées dans le tenseur. Ce que nous avons créé ci-dessus est un tenseur de rang `0`, également connu sous le nom de **scalaire**.<br/>

Maintenant, nous allons créer un tenseur de plus haut degré/rang.

```python
rank1_tensor = tf.Variable(['Test', 'ok'], tf.string)
rank2_tensor = tf.Variable([['Test', 'ok'], ['test', 'yes']], tf.string)

```

Pour déterminer le rang d'un tenseur, tu peux appeller la méthod `tf.rank()` comme suit:

```python
tf.rank(rank2_tensor)

```

### Dimension des tenseurs
Maintenant que nous avons parlé du rang des tenseurs, il est temps de parler de leur forme. La forme d'un tenseur est simplement la quantité d'éléments qui existent dans chaque dimension. TensorFlow essaiera de déterminer la dimension d'un tenseur mais parfois elle peut être inconnue.<br/>

Pour obtenir les dimension d'un tenseur, on utilise l'attribut `shape`.

```python
rank2_tensor.shape  # Retourne un tuple

```

### Redimensionnement d'un tenseur
Le nombre d'éléments d'un tenseur est le produit des valeurs `x1`, `x2`, ..., `xi`, ... `xn`, de toutes ses dimensions `(x1, x2, ..., xi, ..., xn)`. Etant donnée que deux tenseurs de dimensions différentes peuvent contenir le même nombre d'éléments, alors il est possible de changer les dimensions d'un tenseur. <br/>

L'exemple ci-dessous montre comment changer les dimensions d'un tenseur.

```python
tensor1 = tf.ones([1, 2, 3])  # Crée un tenseur de dimensions (1, 2, 3) remplit de 1.
tensor2 = tf.reshape(tensor1, [2, 3, 1])  # Redimensionne le tenseur existant en (2, 3, 1).
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 indique au tenseur de calculer 
                                        # la taille de la dimension à cet endroit. 
                                        # Ce qui donnera au tenseur la forme [3, 3].

# Le nombre d'éléments dans le tenseur redimensionné DOIT correspondre 
# au nombre d'éléments dans le tenseur d'origine.

```

Maintenant, jetons un coup d'oeil à nos différents tenseurs.

```python
print(f"{tensor1 = }\n")
print(f"{tensor2 = }\n")
print(f"{tensor3 = }\n")

# Remarque les changements de dimension.

```

```
tensor1 = <tf.Tensor: shape=(1, 2, 3), dtype=float32, numpy=
array([[[1., 1., 1.],
        [1., 1., 1.]]], dtype=float32)>

tensor2 = <tf.Tensor: shape=(2, 3, 1), dtype=float32, numpy=
array([[[1.],
        [1.],
        [1.]],

       [[1.],
        [1.],
        [1.]]], dtype=float32)>

tensor3 = <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[1., 1.],
       [1., 1.],
       [1., 1.]], dtype=float32)>

```

### Types de tenseurs
Avant que nous Ce sont les plus couramment utilisés et nous parlerons plus en profondeur de chacun d'eux au fur et à mesure de leur utilisation.

- `Variable`;
- `Constant`;
- `Placeholder`;
- `SparseTensor`.

À l'exception de `Variable`, tous ces tenseurs sont immuables, ce qui signifie que leur valeur ne peut pas changer pendant l'exécution. Pour l'instant, il suffit de comprendre que nous utilisons le tenseur `Variable` lorsque nous voulons potentiellement changer la valeur de notre tenseur.


### Evaluation des tenseurs
Parfois, on a besoin d'évaluer un tenseur. En d'autres termes, récupérer sa valeur. Comme les tenseurs représentent un calcul partiellement complet, nous devrons parfois exécuter ce que l'on appelle une **session** pour évaluer le tenseur. <br/>

Il existe de nombreuses manières différentes d'y parvenir. Mais je vais noter ci-dessous la manière la plus simple.

```python
# On crée la session en utilisant le graphe par défaut:
with tf.compat.v1.Session() as sess: 
    tensor1.eval()

```

Ou tout simplement, faire comme ceci:

```python
tf.print(tensor1)

```

Dans le code ci-dessus, nous avons évalué la variable tensorielle qui était stockée dans le *graphe par défaut*. Le graphe par défaut contient toutes les opérations qui ne sont pas spécifiées dans un autre graphe. Il est possible de créer nos propres graphes séparés. Mais pour l'instant, nous allons nous en tenir au graphe par défaut.

<br/>
<br/>

- --> Je passe à la session **suivante**: [Algorithmes d'apprentissage fondamentaux](./core_learning_algorithms/README.md)
- <-- Je reviens à la session **précédente**: [Accueil](../README.md)
