# Entrenamiento de modelos de deteccion de objetos con YoloV11

Este repositorio contiene un script para entrenar modelos de detección de objetos con YoloV11.

## Construido con 🛠️

- [Python v3.9](https://www.python.org/) - Lenguaje de programación
- [Ultralytics](https://ultralytics.com/) - Librería de modelo de detección de objetos

## Comenzando 🚀

### Instalacion  🔧

<details>
   <summary>Linux</summary>

1. Se debe instalar venv
    ```bash
    sudo apt-get install python3.9-venv
    ```

3. Se clona el repositorio de GitHub
    ```bash
    git clone https://github.com/ViceAguilera/Train-YoloV11-Model.git
    ```
  
4. Se ingresa a la carpeta del proyecto
    ```bash
    cd Train-YoloV11-Model
    ```
  
5. Se crea un entorno virtual
    ```bash
    python3.9 -m venv venv
    ```
    
6. Se activa el entorno virtual
    ```bash
    source venv/bin/activate
    ```

7. Se instala los requerimientos del proyecto
    ```bash
    pip install ultralytics
    ```
   
8. Se desinstala pytorch
    ```bash
    pip uninstall -y torch torchvision torchaudio
    ```

8. Se instala CUDA Pytorch
    ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
</details>

<details>
  <summary>Windows</summary>

1. Se clona el repositorio de GitHub
    ```bash
    git clone https://github.com/ViceAguilera/Train-YoloV11-Model.git
    ```
   
2. Se ingresa a la carpeta del proyecto
    ```bash
    cd Train-YoloV11-Model
    ```
   
3. Se crea un entorno virtual
    ```bash
    python -m venv venv
    ```
   
4. Se activa el entorno virtual
    ```bash
    .\venv\bin\activate
    ```

5. Se instala los requerimientos del proyecto
    ```bash
    pip install ultralytics
    ```
   
6. Se desinstala pytorch
    ```bash
    pip uninstall -y torch torchvision torchaudio
    ```

7.  Se instala CUDA Pytorch
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

</details>

### Configuración ⚙️

- Se debe cambiar el `path` en el archivo `config.yaml` en la linea 1, para escribir la ubicacion de la carpeta Train-model-yolov8.
- Se debe guardar en las carpetas `test` , `train` y `valid` la carpeta `images` con las imagenes y la carpeta `labels`  los archivos `.txt` con las coordenadas de los objetos a detectar. (Se recomienda ocupar Roboflow)

### Ejecución 📸
*Python*

    python train.py

*CLI*

    yolo task=detect mode=train epochs=150 data=config.yaml model=yolov8n.pt imgsz=640 batch=16

> Este comando esta sujeto a cambios según lo que usted estime conveniente.
 
## Licencia 📄

Este proyecto está bajo el _MIT LICENSE_ - mira el archivo [LICENSE](LICENSE) para detalles

## Autor ✒️

[**Vicente Aguilera Arias**](https://github.com/ViceAguilera)