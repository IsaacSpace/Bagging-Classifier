from sklearn.base import BaseEstimator
import numpy 

class BaggingClassifier:
    """
    Implementación de un clasificador Bagging.

    Args:
        model: El modelo base que se utilizará en cada estimador.
            Debe ser un objeto que implemente el método .fit() para entrenar el modelo.
        n_estimators: Número de estimadores en el conjunto (bolsa).
            Debe ser un entero mayor que 0.

    Attributes:
        model: El modelo base utilizado en los estimadores.
        n_estimators: Número de estimadores en el conjunto.
        fitted_models: Lista que almacena los modelos entrenados.

    Methods:
        fit(X, y): Entrena los estimadores en el conjunto.
        predict(X): Realiza predicciones utilizando los estimadores en el conjunto.
    """

    def __init__(self, model, n_estimators):
        """
        Inicializa un clasificador Bagging.

        Args:
            model: El modelo base que se utilizará en cada estimador.
                Debe ser un objeto que implemente el método .fit() para entrenar el modelo.
            n_estimators: Número de estimadores en el conjunto (bolsa).
                Debe ser un entero mayor que 0.

        """
        if not isinstance(model, BaseEstimator):
            raise ValueError("El argumento 'model' debe ser un objeto de scikit-learn.")
        self.model = model 
        self.n_estimators = n_estimators

    def fit(self, X, y):
        """
        Entrena los estimadores en el conjunto.

        Args:
            X: Conjunto de datos de entrenamiento.
                Debe ser un array numpy o una estructura similar compatible con scikit-learn.
            y: Etiquetas de clases correspondientes al conjunto de datos.
                Debe ser un array numpy u otra estructura compatible con scikit-learn.
        """
        self.fitted_models = [self.model for i in range(0, self.n_estimators, 1)]
        for i in range(1, self.n_estimators+1, 1):
            print(f"Fitting model {i}")
            self.fitted_models[i-1].fit(X, y)
    
    def predict(self, X):
        """
        Realiza predicciones utilizando los estimadores en el conjunto.
        Args:
            X: Conjunto de datos de prueba.
                Debe ser un array numpy o una estructura similar compatible con scikit-learn.

        Returns:
            predictions: Predicciones resultantes.
                Un array numpy con las predicciones correspondientes.
        """
        n_vectors, n_features = X.shape
        results = numpy.zeros(shape=(n_vectors, self.n_estimators))
        predictions = numpy.zeros(shape=n_vectors)
        for j in range(0, self.n_estimators, 1):
            for i in range(0, n_vectors, 1):
                vector = X[i, :].reshape(-1, 1)
                results[i, j] = self.fitted_models[j].predict(vector.T)
        for j in range(0, n_vectors, 1):
            predictions[j] = numpy.unique(results[j, :], return_counts=True)[0]
        return predictions
    


