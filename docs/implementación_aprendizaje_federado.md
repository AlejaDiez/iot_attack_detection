# Implementación de Aprendizaje Federado

Para la implementación del aprendizaje federado existen diversas herramientas y frameworks que facilitan la implementación de este modelo de aprendizaje. A continuación, muestro los dos frameworks en los que me he basado para la implementación de este modelo de aprendizaje.

## TensorFlow Federated

[TensorFlow Federated](https://www.tensorflow.org/federated) es un framework de python de código abierto desarrollado y mantenido por Google. Permite entrenar modelos de aprendizaje automático generados en TensorFlow, mediante la distribución de los datos y el cómputo en múltiples dispositivos. TensorFlow Federated proporciona dos APIs de alto nivel para el desarrollo de aplicaciones de aprendizaje federado:

-   **Federated Core**: API que permite implementar computaciones distribuidas en TensorFlow Federated. Cada una de las computaciones realiza tareas complejas y permite la realizar de una forma ordenada la comunicación entre ellas.
-   **Federated Learning**: API de alto nivel que permiten adaptar modelos de aprendizaje automático existentes a TensorFlow Federated. Esta API se basa en la API de Federated Core. Con esta API se pueden implementar algoritmos de aprendizaje federado y se divide en tres partes fundamentales:
    -   **Modelos**: clases y funciones que permiten convertir modelos existentes a modelos de computación federativa.
    -   **Constructores de operaciones federativas**: funciones que permiten construir operaciones federativas.
    -   **Conjuntos de datos**: grupo de datos para realizar escenarios de simulación.

## Flower

[Flower](https://flower.ai/) es un framework de aprendizaje federado de código abierto que permite adaptar modelos de aprendizaje automático a escenarios de aprendizaje federado. Flower permite usar cualquier marco de aprendizaje automático y proporciona una API de alto nivel para el desarrollo de aplicaciones de aprendizaje federado. Flower se puede usar con TensorFlow, PyTorch, Keras, entre otros. Flower proporciona las siguientes características:

-   **Personalizable**: Flower permite personalizar el aprendizaje federado para adaptarse a diferentes casos de uso.
-   **Extensible**: Flower se originó en un proyecto de investigación en la Universidad de Oxford, por lo que fue construido con la investigación en IA en mente. Muchos componentes pueden ser extendidos y anulados para construir nuevos sistemas.
-   **Agnóstico al marco de trabajo**: Flower se puede usar con cualquier framework de aprendizaje automático, por ejemplo, PyTorch, TensorFlow, Hugging Face Transformers, PyTorch Lightning, scikit-learn, JAX...
-   **Comprensible**: Flower está escrito con la mantenibilidad en mente. Se anima a la comunidad a leer y contribuir al código.

## Bibliografía

-   [Top 7 Open-Source Frameworks for Federated Learning](https://www.apheris.com/resources/blog/top-7-open-source-frameworks-for-federated-learning)
