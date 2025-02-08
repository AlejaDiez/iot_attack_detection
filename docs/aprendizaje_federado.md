# Aprendizaje Federado

## Introducción

Cuando se trata de entrenar modelos de aprendizaje automático, los datos son un punto a tener en cuenta. En la mayoría de los casos, los datos se almacenan en un servidor centralizado, donde se van a utilizar para entrenar el modelo. Sin embargo, existen casos donde los datos son sensibles y no se pueden compartir con otras organizaciones externas, como en el caso de los datos médicos donde la privacidad de los pacientes esta por encima de todo o en el caso de los datos financieros de una empresa. Para estos casos, el aprendizaje federado puede ser una solución para entrenar modelos de aprendizaje automático sin tener que extraer los datos de su ubicación original.

## ¿Qué es el aprendizaje federado?

El aprendizaje federado es un enfoque de aprendizaje automático que permite entrenar un modelo de aprendizaje automático en diferentes dispositivos que contienen datos locales y que estos no pueden ser extraidos de su ubicación original, siguiendo una arquitectura descentralizada. En lugar de enviar los datos a un servidor central y que este se encargue de entrenar el modelo, el modelo se envía a los diferentes dispositivos que forman parte de la red de aprendizaje federado y se entrena en cada uno de ellos. Una vez que el modelo ha sido entrenado en cada dispositivo, los pesos del modelo se envían de vuelta al servidor central, donde se combinan con los pesos de los otros dispositivos para formar un modelo global. Este modelo global se vuelve a enviar a los dispositivos para que puedan seguir entrenando el modelo con sus datos locales y así se realizará un ciclo de entrenamiento hasta que el modelo converja.

## Tipos de aprendizaje federado

Según el tipo de datos que se estén utilizando, el aprendizaje federado se puede dividir en tres tipos:

- **Aprendizaje federado horizontal**: este tipo de aprendizaje federado usa datos con características similares, pero de diferentes usuarios. Por ejemplo, el entrenamiento de un modelo con transacciones bancarias del mismo tipo, pero de diferentes clientes.
- **Aprendizaje federado vertical**: este tipo de aprendizaje federado usa datos con características diferentes para entrenarlas en conjunto y obtener un modelo global. Por ejemplo, el entrenamiento de un modelo de recomendación de peliculas, donde se usa la información de peliculas compradas en una plataforma y la información de reseñas de los usuarios sobre las peliculas en otra plataforma.
- **Aprendizaje federado por transferencia**: este tipo de aprendizaje federado usa un modelo pre-entrenado en un conjunto de datos y se entrena con otro conjunto de datos similar para solucionar otro tipo de problema. Por ejemplo, el entrenamiento de un modelo de reconocimiento de imagenes para detectar imagenes de animales y se entrena con otro conjunto de datos de imagenes de animales para reconocer diferentes especies.

## Funcionamiento del aprendizaje federado

El entrenamiento de un modelo de aprendizaje federado se puede dividir en los siguientes pasos:

1. **Inicialización del modelo**: se inicializa un modelo de aprendizaje automático en el servidor central.
2. **Envío del modelo a los dispositivos**: el modelo se envía a los diferentes dispositivos que forman parte de la red de aprendizaje federado.
3. **Entrenamiento del modelo en cada dispositivo**: el modelo se entrena en cada dispositivo con los datos locales que contienen.
4. **Envío de los pesos del modelo al servidor central**: una vez que el modelo ha sido entrenado en cada dispositivo, los pesos del modelo se envían de vuelta al servidor central.
5. **Combinación de los pesos del modelo**: los pesos del modelo se combinan con los pesos de los otros dispositivos mediante un algoritmo para formar un modelo global.
6. **Envío del modelo global a los dispositivos**: el modelo global se envía a los dispositivos para que puedan seguir entrenando el modelo con sus datos locales.
7. **Ciclo de entrenamiento**: se repiten los pasos 3 a 6 hasta que el modelo converja.

## Bibliografía

- [Federated Learning Comic](https://federated.withgoogle.com)
- [What is Federated Learning?](https://www.youtube.com/watch?v=X8YYWunttOY)
- [TensorFlow Federated Tutorial Session](https://www.youtube.com/watch?v=JBNas6Yd30A&t=10551s)
- [Aprendizaje Federado](https://es.wikipedia.org/wiki/Aprendizaje_federado)
- [Introduction to Federated Learning and Challenges](https://towardsdatascience.com/introduction-to-federated-learning-and-challenges-ea7e02f260ca)
