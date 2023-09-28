# BaggingClassifier

El `BaggingClassifier` es una implementación de un clasificador Bagging en Python. El clasificador Bagging es una técnica de ensamblaje que entrena múltiples modelos de manera independiente y luego combina sus predicciones para mejorar la precisión y la generalización.

## Uso

Puedes utilizar la clase `BaggingClassifier` para entrenar un conjunto de modelos base y realizar predicciones mediante el método de Bagging. A continuación, se muestra cómo utilizar la clase:

```python
from BaggingClassifier import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Crear un clasificador base (por ejemplo, un árbol de decisión)
base_model = DecisionTreeClassifier()

# Crear un clasificador Bagging con 5 estimadores
bagging = BaggingClassifier(base_model, n_estimators=5)

# Entrenar el clasificador con datos de entrenamiento (X_train, y_train)
bagging.fit(X_train, y_train)

# Realizar predicciones con datos de prueba (X_test)
predictions = bagging.predict(X_test)
