Predicción de Imágenes con ResNet50 y Tkinter
Este programa implementa un reconocedor de imágenes utilizando el modelo ResNet50 pre-entrenado en el conjunto de datos ImageNet y la biblioteca Tkinter para la interfaz gráfica de usuario.

Descripción
La aplicación permite a los usuarios cargar una imagen desde su dispositivo, procesarla utilizando el modelo ResNet50 y mostrar la predicción resultante, que incluye el objeto reconocido y su probabilidad asociada.

Funcionalidades
Cargar Imagen: Permite al usuario seleccionar una imagen desde su dispositivo.
Generar Predicción: Utiliza el modelo ResNet50 para realizar una predicción sobre la imagen cargada.
Restaurar: Limpia la interfaz para restablecer el programa a su estado original.
Tecnologías Utilizadas
Tkinter: Para la creación de la interfaz gráfica de usuario.
PIL (Python Imaging Library): Para el procesamiento de imágenes.
NumPy: Para la manipulación de arrays.
TensorFlow y Keras: Para utilizar el modelo pre-entrenado ResNet50.
Uso
Para ejecutar la aplicación, simplemente ejecute el script en Python. A continuación, podrá cargar una imagen haciendo clic en el botón "Cargar Imagen". Una vez cargada la imagen, haga clic en "Generate Prediction" para realizar una predicción sobre la misma.

Instalación
Clonar este repositorio:


git clone https://github.com/tu_usuario/nombre_del_repositorio.git

python main.py en consola
Contribuciones
Las contribuciones son bienvenidas. Si desea mejorar este proyecto, no dude en enviar un pull request.

Licencia
Este proyecto está bajo la Licencia MIT. Consulte el archivo LICENSE para obtener más detalles.
