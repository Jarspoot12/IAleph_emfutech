# IAleph
Este proyecto contiene la estructura del programa IAleph, el cual está diseñado para ejecutar tareas de visión por computadora aplicada a la identificación de clientes.

Para ejecutar el programa que implementa el uso de dos cámaras ejecuta main3.py, que se ejecuta con el tracker de deepsort y deepface.

Se cree que existe un problema con la generación de embeddings, ya que esto es lo que debe permitir el re ID tracking incluso de una cámara a otra.


Para mi uso personal:
1. Actualmente estoy implementando un modelo con fine-tuning para ser usado al momento de detectar el género, pues se presentan problemas con el accuracy al usar un modelo preentrenado.
2. Estoy usando el tracking de deepface y deepsort con la lógica del re id tracking, que es una alternativa al archivo tracker_fastreid.py, el cual se sospecha presenta problemas al crear los embeddings.
3. Todo el flujo de trabajo se concentra en el main, donde importo los módulos anteriores para ser usados en mi pipeline.

## Modelos
Este es el enlace donde se encuentran los modelos usados: https://drive.google.com/drive/folders/1ULuiaeUdWgiIl0iKiQA1FXQEH1cU40YA?usp=sharing


