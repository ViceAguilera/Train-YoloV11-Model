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
- Se debe guardar en las carpetas `test` , `train` y `valid` la carpeta `images` con las imagenes y la carpeta `labels`  los archivos `.txt` con las coordenadas de los objetos a detectar.

### Ejecución 📸
*Python*

    python train.py

*CLI*

    yolo task=detect mode=train epochs=150 data=config.yaml model=yolov8n.pt imgsz=640 batch=16

> Este comando esta sujeto a cambios según lo que usted estime conveniente.
 
## Dataset 📦

El conjunto de datos utilizado para el entrenamiento de los modelos de detección de EPP fue obtenido de [Roboflow](https://roboflow.com/), específicamente del proyecto [PPE Detection Using CV Computer Vision Project](https://universe.roboflow.com/ai-camp-safety-equipment-detection/ppe-detection-using-cv).
## Licencia 📄

Este proyecto está bajo el _MIT LICENSE_ - mira el archivo [LICENSE](LICENSE) para detalles

## Autor ✒️

[**Vicente Aguilera Arias**](https://github.com/ViceAguilera)

## Análisis de entrenamientos

Para cada entrenamiento, Ultralytics guarda las métricas en `runs/<nombre_run>/results.csv` junto con gráficos (`results.png`, `PR_curve.png`, `confusion_matrix.png`, etc.).

Este repositorio incluye un script para resumir todos los entrenamientos realizados:

```bash
python analyze_runs.py
```

El script recorre la carpeta `runs/` y muestra por cada run:
- modelo utilizado (`model`)
- tamaño de imagen (`imgsz`)
- batch size y épocas (`batch`, `epochs`)
- mejor `mAP50` y `mAP50-95` (y en qué época ocurre)
- métricas finales del último epoch

### Historial resumido (train1–12 y tuned)

Resumen aproximado de los entrenamientos más importantes (valores redondeados):

| Run               | Modelo     | Img | Épocas | Best mAP50-95 | Best mAP50 | Época best | Final mAP50-95 | Final mAP50 |
|-------------------|-----------:|----:|-------:|--------------:|-----------:|-----------:|---------------:|------------:|
| `train`           | yolo11n.pt | 640 |    150 |         0.417 |      0.745 |         78 |          0.404 |       0.741 |
| `train3`          | yolo11n.pt | 640 |    300 |         0.206 |      0.389 |          4 |          0.206 |       0.389 |
| `train4`          | yolo11n.pt | 640 |    300 |         0.306 |      0.517 |         89 |          0.259 |       0.428 |
| `train5`          | yolo11n.pt | 640 |    100 |         0.283 |      0.476 |         77 |          0.270 |       0.456 |
| `train6`          | yolo11n.pt | 640 |    150 |         0.307 |      0.519 |         46 |          0.256 |       0.425 |
| `train11`         | yolo11n.pt | 640 |    150 |         0.516 |      0.882 |         54 |          0.494 |       0.862 |
| `train12`         | yolo11n.pt | 640 |    150 |         0.530 |      0.884 |         71 |          0.515 |       0.876 |
| `train_tuned2`    | yolo11s.pt | 800 |    180 |         0.504 |      0.869 |        166 |          0.495 |       0.864 |
| `train_tuned_v22` | yolo11s.pt | 640 |    160 |         0.506 |      0.865 |        127 |          0.496 |       0.852 |
| `train_tuned_v3`  | yolo11s.pt | 640 |    136*|         0.509 |      0.879 |        106 |          0.496 |       0.867 |

Notas rápidas:
- El mejor resultado global actual es `train12` (mAP50-95 ≈ 0.53, mAP50 ≈ 0.88).
- Los runs `train_tuned2` y `train_tuned_v22` usan `yolo11s.pt` y se acercan al rendimiento de `train12`, pero no lo superan todavía.
- Los runs `train2`, `train7`, `train8`, `train9` y `train10` no tienen `results.csv`, por lo que no se incluyen en la tabla.
