# IoT Simulator

Este directorio contiene el proyecto del Simulador IoT, que se encarga de emular el comportamiento de una red de dispositivos IoT para la detección de ataques mediante inteligencia artificial.

## Requisitos

Para poder ejecutar el simulador, es necesario tener al menos Node.js 18.19.1 o superior y angular 19.0.0 o superior instalados en tu sistema.

Puedes instalar angular CLI globalmente utilizando el siguiente comando:

```bash
npm install -g @angular/cli
```

Para instalar las dependencias necesarias del proyecto, ejecuta:

```bash
npm install
```

## Ejecución

Para ejecutar el simulador, utiliza el siguiente comando:

```bash
npm run start
```

Esto iniciará el servidor de desarrollo y podrás acceder al simulador en tu navegador web en la dirección `http://localhost:4200`.

## Construcción

Si deseas construir el proyecto para producción, puedes utilizar el siguiente comando:

```bash
npm run build
```

Esto generará los archivos de producción en el directorio `dist/` del proyecto, que podrás desplegar en un servidor web. Es necesario que cambies la URL base del servidor en los scripts del archivo `package.json` antes de construir el proyecto, para que apunte a la dirección correcta del servidor donde se desplegará el simulador.
