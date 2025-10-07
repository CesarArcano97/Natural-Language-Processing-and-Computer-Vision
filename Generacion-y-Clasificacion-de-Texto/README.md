- # Tarea 2: Generación y Clasificación de Texto con Deep Learning

  ## Descripción General

  Este repositorio contiene el código y los resultados de la Tarea #2 para la materia de Procesamiento de Lenguaje Natural y Visión Computacional. El proyecto explora y compara diversas arquitecturas de Deep Learning para dos tareas fundamentales del NLP:

  1.  **Parte A - Generación de Texto**: Se entrenaron modelos para generar letras de canciones que imitan el estilo de la banda *Twenty One Pilots*.
  2.  **Parte B - Clasificación de Texto**: Se entrenaron modelos para clasificar la polaridad (sentimiento en una escala de 1 a 5 estrellas) de reseñas turísticas del dataset MeIA.

  

  ## Objetivos

  Los objetivos principales de este proyecto, de acuerdo con las especificaciones de la tarea, fueron:

  * **Construir modelos generativos** de letras de canciones (RNN/LSTM/GRU/Transformers) y evaluar su calidad.
  * **Entrenar modelos de clasificación** de texto (CNN/RNN/LSTM/GRU/Transformers) sobre el dataset utilizado en la Tarea #1.
  * **Realizar fine-tuning** de al menos un modelo Transformer para cada una de las dos partes.
  * **Analizar y comparar** las métricas de rendimiento y los costos computacionales de los distintos enfoques.

  ---

  ## 🛠️ Arquitecturas Exploradas

  Se implementaron y evaluaron las siguientes arquitecturas:

  * **Generación de Texto (Parte A)**:
      * Red Neuronal Recurrente (RNN)
      * Long Short-Term Memory (LSTM)
      * Gated Recurrent Unit (GRU)
      * Transformer (fine-tuning de `TinyLLaMA-1.1B`)
      * Transformer (fine-tuning de `Mistral-7B` con LoRA)

  * **Clasificación de Texto (Parte B)**:
      * Convolutional Neural Network (TextCNN)
      * Red Neuronal Recurrente Bidireccional (Bi-RNN)
      * Bidirectional LSTM (BiLSTM)
      * Bidirectional GRU (BiGRU)
      * *(Pendiente: Transformer con fine-tuning, ej. BETO/DistilBERT)*

  ---

  ## Dataset

  * **Para la Generación (Parte A)**: Se creó un corpus personalizado con más de 100 letras de canciones de la banda Twenty One Pilots. La recolección de datos se realizó utilizando la API de Genius.
  * **Para la Clasificación (Parte B)**: Se utilizó el dataset de reseñas turísticas en español (MeIA) de la Tarea #1, con el objetivo de clasificar la polaridad en 5 clases.

  ---

  ## Entorno Computacional y Metodología de Fine-Tuning

  El fine-tuning de los modelos Transformer (`TinyLLaMA-1.1B` y `Mistral-7B`) se llevó a cabo en el clúster del **Laboratorio de Supercómputo del Bajío (Lab-SB)** del CIMAT, debido a los altos requerimientos de VRAM y capacidad de cómputo.

  ### Hardware y Software

  * **Clúster**: Lab-SB (host: `el-insurgente.cimat.mx`).
  * **GPU**: Nodos equipados con GPUs NVIDIA TITAN RTX (24 GB de VRAM).
  * **Gestor de Trabajos**: SLURM para la administración de colas y recursos.
  * **Entorno**: Miniconda para la gestión de entornos de software aislados.
  * **Librerías Clave**: PyTorch, Transformers, `unsloth` para optimización de memoria (QLoRA), `peft` para LoRA, y `bitsandbytes`.

  ### Flujo de Trabajo (Workflow)

  Una característica clave del clúster es que los nodos de cómputo **no tienen acceso a internet**. Esto requirió un flujo de trabajo completamente offline:

  1.  **Preparación Local**: Los modelos base (ej. `Mistral-7B-Instruct-v0.2`), tokenizers y datasets se descargaron previamente en una máquina local.
  2.  **Transferencia de Datos**: Todos los artefactos necesarios (modelos, código fuente, datasets) se transfirieron al directorio `$HOME` del servidor mediante `scp`.
  3.  **Ejecución con SLURM**: Se crearon scripts de bash (`.sh`) o SLURM (`.slurm`) para definir los recursos y lanzar los trabajos de entrenamiento y generación de forma no interactiva con el comando `sbatch`.
  4.  **Entrenamiento Eficiente**: Se utilizaron técnicas de LoRA (a través de `peft`) y QLoRA (cuantización de 4-bits con `unsloth`) para poder ajustar un modelo de 7 mil millones de parámetros en una única GPU de 24 GB, entrenando menos del 1% de los parámetros totales.
  5.  **Monitoreo y Resultados**: Los trabajos se monitorearon con `squeue` y `tail -f`, y los artefactos resultantes (adaptadores LoRA, logs y textos generados) se descargaron de vuelta a la máquina local para su análisis.

  ### Prerrequisitos

  * Python 3.8+
  * PyTorch 2.0+
  * Git

  ### Uso

  Los scripts para entrenar y evaluar los modelos se encuentran en el directorio `src/`.

  **Ejemplo: Entrenar un clasificador BiLSTM**
  ```bash
  python -m src.train.train_rnnclf \
    --data_dir data/processed/classif/five \
    --out_dir models/bilstm/exp_bilstm_baseline \
    --rnn_type lstm \
    --bidirectional \
    --embed_dim 256 --hidden_size 256 \
    --num_layers 1 \
    --pool max \
    --proj_dropout 0.5 \
    --lr 1e-3 \
    --batch_size 64 --epochs 25 --patience 5
  ```

  **Ejemplo: Evaluar un modelo guardado**
  ```bash
  python -m src.eval.evaluate_rnn \
    --ckpt models/bilstm/exp_bilstm_baseline/best_model.pt \
    --data_dir data/processed/classif/five \
    --out_dir models/bilstm/exp_bilstm_baseline
  ```

  ---

  ## Resumen de Resultados

  El análisis completo, las gráficas y las conclusiones detalladas se encuentran en el **reporte en PDF**. A continuación, un resumen de los hallazgos clave:

  ### Generación de Texto

  * Los modelos Transformer, especialmente **Mistral-7B con LoRA**, demostraron una superioridad abrumadora en la calidad de la generación. Produjeron texto con mayor coherencia, estructura lírica y fidelidad estilística.
  * Las arquitecturas recurrentes (RNN, LSTM, GRU) sirvieron como excelentes líneas base, pero mostraron limitaciones en la coherencia a largo plazo y una mayor tendencia a la repetición.

  ### Clasificación de Texto

  * Se estableció una clara jerarquía de rendimiento entre las arquitecturas no-Transformers, donde la **BiLSTM** obtuvo el mejor resultado:
      1.  **BiLSTM** (F1-Macro: 0.515) 
      2.  **BiGRU** (F1-Macro: 0.491) 
      3.  **Bi-RNN** (F1-Macro: 0.484) 
      4.  **TextCNN** (F1-Macro: 0.441)
  * Un desafío constante en todos los modelos fue la clasificación de las clases intermedias y ambiguas (2, 3 y 4 estrellas), mientras que las clases extremas (1 y 5) fueron identificadas con mayor facilidad.

  ---

  ## 📁 Estructura del Proyecto

  ```
  .
  ├── data/                  # Scripts para descargar y procesar datos
  ├── models/                # Checkpoints de modelos entrenados y resultados
  │   ├── cnn/
  │   ├── rnn/
  │   └── ...
  ├── report/
  │   └── Tarea2_Reporte.pdf   # Reporte final del proyecto
  ├── src/
  │   ├── data/              # Módulos de carga de datos (Dataset, Dataloader)
  │   ├── eval/              # Scripts para evaluar modelos
  │   ├── models/            # Definiciones de arquitecturas de modelos
  │   └── train/             # Scripts para entrenar modelos
  ├── requirements.txt       # Dependencias del proyecto
  └── README.md              # Este archivo
  ```

  ---

  ## Autor

  * César M. Aguirre Calzadilla

  

  ## Licencia

  Este proyecto está bajo la Licencia MIT.

