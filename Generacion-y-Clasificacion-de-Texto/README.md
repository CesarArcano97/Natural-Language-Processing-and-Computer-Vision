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

  ## Arquitecturas Exploradas

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

  

  # Proyecto de Modelos Generativos y Clasificatorios – Tarea 02  
  **Procesamiento de Lenguaje Natural (PLN)**  
  **Autor:** *César Aguirre*  
  **Servidor de entrenamiento:** CIMAT HPC Cluster  
  
  ---
  
  ## Objetivos generales
  
  El proyecto aborda dos grandes bloques complementarios dentro del PLN:
  
  1. **Generación de texto:**  
     Entrenar modelos generativos (RNN, LSTM, GRU y Transformer finetuneado) para producir letras de canciones.
  
  2. **Clasificación de texto:**  
     Entrenar modelos de clasificación (CNN, RNN, LSTM, GRU y BETO) sobre el mismo corpus, con el fin de comparar desempeño, arquitectura y comportamiento semántico.

  ---
  
  ## Estructura del proyecto
  
  ```bash
  project/
  │
  ├── data/
  │   ├── raw/
  │   │   ├── canciones.txt
  │   │   └── MeIA.csv
  │   └── processed/
  │       ├── char/
  │       ├── word/
  │       └── classif/five/
  │
  ├── models/
  │   ├── rnn/
  │   ├── lstm/
  │   ├── gru/
  │   ├── cnn/
  │   ├── transformer/         # Fine-tuning Mistral / BETO
  │   └── checkpoints/
  │
  ├── outputs/
  │   ├── sample_rnn.txt
  │   ├── sample_lstm.txt
  │   ├── sample_gru.txt
  │   └── mistral_output.txt
  │
  ├── results/
  │   └── novelty/
  │       ├── novelty_analysis.csv
  │       └── novelty_plot.png
  │
  ├── scripts/
  │   ├── generate_mistral_offline.slurm
  │   └── train_beto.slurm
  │
  ├── src/
  │   ├── prepare_corpus.py
  │   ├── generate.py
  │   ├── analyze_novelty.py
  │   ├── train_rnn.py
  │   ├── train_lstm.py
  │   ├── train_gru.py
  │   ├── train_cnn.py
  │   ├── train_rnnclf_01.py
  │   ├── eval/
  │   │   ├── evaluate_rnn.py
  │   │   └── evaluate.py
  │   └── transformers/
  │       └── fine_tune_beto.py
  │
  └── requirements.txt
  ```
  
  ________________________________
  
  ## Dependencias
  
  El entorno de trabajo requiere **Python 3.9+** y las siguientes librerías principales:
  
  ```
  torch
  transformers
  peft
  tqdm
  numpy
  pandas
  matplotlib
  scikit-learn
  ```
  
  ### Instalación
  
  ```
  conda create -n nlp-t02 python=3.9
  conda activate nlp-t02
  pip install -r requirements.txt
  ```
  
  ---
  
  ## PARTE I — Modelos Generativos
  
  ### Objetivo
  
  Generar letras de canciones en el estilo de *Twenty One Pilots*, explorando arquitecturas secuenciales clásicas (RNN, LSTM, GRU) y modelos de lenguaje grandes (Mistral).
  
  ------
  
  ### Flujo de trabajo
  
  #### 1. Preparar corpus
  
  ```
  python src/prepare_corpus.py --input data/raw/canciones.txt --level char
  python src/prepare_corpus.py --input data/raw/canciones.txt --level word
  ```
  
  #### 2. Entrenar modelos
  
  ```
  python src/train_rnn.py --level word --epochs 20
  python src/train_lstm.py --level word --epochs 20
  python src/train_gru.py --level char --epochs 20
  ```
  
  #### 3. Generar letras
  
  ```
  python src/generate.py --model models/lstm/best_model.pt \
    --prompt "In Dema there's no choice, but in Trench I'm not afraid of"
  ```
  
  #### 4. Evaluar con métricas
  
  ```
  python src/analyze_novelty.py \
    --corpus data/raw/canciones.txt \
    --outputs outputs/sample_rnn.txt outputs/sample_lstm.txt outputs/sample_gru.txt \
    --out_dir results/novelty
  ```
  
  ------
  
  ### Fine-tuning con Mistral (Transformer)
  
  Ejecutado en el servidor de CIMAT:
  
  ```
  conda activate mistral-env
  python src/generate/test_generate.py
  ```
  
  Ejemplo:
  
  ```
  prompt = "Write a song in the style of Twenty One Pilots about self-discovery"
  ```
  
  O mediante SLURM:
  
  ```
  sbatch scripts/generate_mistral_offline.slurm
  ```
  
  | Modelo       | PPL ↓ | Novelty ↑ (Bigram) | Novelty ↑ (Trigram) | Observaciones                 |
  | ------------ | ----- | ------------------ | ------------------- | ----------------------------- |
  | RNN          | Alto  | 57.49%             | 78.16%              | Tiende a memorizar            |
  | LSTM         | Bajo  | 65.58%             | 92.16%              | Mejor equilibrio              |
  | GRU          | Medio | 62.32%             | 92.72%              | Coherencia similar            |
  | Mistral (FT) | Bajo  | —                  | —                   | Estilo y coherencia mejoradas |
  
  
  
  ## PARTE II — Modelos de Clasificación
  
  ### Objetivo
  
  Clasificar fragmentos de canciones en **5 categorías ordinales** (1–5), explorando diferentes arquitecturas y regularizaciones.
  
  ------
  
  ### Flujo general
  
  #### 1. Preparar dataset
  
  ```
  python -m src.data.prepare_meia \
    --input data/raw/MeIA.csv \
    --out_dir data/processed/classif/five \
    --scheme five --max_len 256 --min_freq 2 --val_size 0.10 --test_size 0.10
  ```
  
  #### 2. Entrenar modelos
  
  **CNN:**
  
  ```
  python -m src.train.train_cnn --data_dir data/processed/classif/five \
    --out_dir models/cnn/exp_textcnn_k345_f128_lr1e-3 \
    --embed_dim 256 --kernel_sizes 3,4,5 --num_filters 128 \
    --lr 1e-3 --weight_decay 1e-4 --proj_dropout 0.5 --epochs 20
  ```
  
  **RNN / LSTM / GRU:**
  
  ```
  python -m src.train.train_rnnclf_01 \
    --data_dir data/processed/classif/five \
    --out_dir models/lstm/exp_bilstm_max_regstrong \
    --rnn_type lstm --embed_dim 256 --hidden_size 256 \
    --num_layers 1 --bidirectional --pool max \
    --emb_dropout 0.3 --rnn_dropout 0.2 --proj_dropout 0.6 \
    --lr 8e-4 --weight_decay 2e-4 --batch_size 64 --epochs 20
  ```
  
  #### 3. Evaluar
  
  ```
  python -m src.eval.evaluate_rnn \
    --data_dir data/processed/classif/five \
    --ckpt models/lstm/exp_bilstm_max_regstrong/best_model.pt \
    --out_dir models/lstm/exp_bilstm_max_regstrong
  ```
  
  ------
  
  ## BETO (Transformer español)
  
  Fine-tuning ejecutado en el **cluster CIMAT**, con un modelo base de **BETO (dccuchile/bert-base-spanish-wwm-cased)**.
   Se adaptó el script `fine_tune_beto.py` para clasificación ordinal (5 clases).
  
  ```
  sbatch scripts/train_beto.slurm
  ```
  
  ------
  
  ## Conclusiones generales
  
  - **Generativos:**
     Las LSTM ofrecen el mejor equilibrio entre coherencia y novedad; los Transformers (Mistral) superan a los modelos clásicos en estilo y consistencia sintáctica.
  - **Clasificatorios:**
     Los modelos secuenciales bi-direccionales (LSTM/GRU) superan a CNNs tradicionales, pero **BETO** logra un salto claro al incorporar contexto global y embeddings preentrenados.
  - **Regularización y pooling** impactan más que la capacidad del modelo:
     Dropout ≈0.5 y `pool=max` resultaron ser los mejores parámetros generales.
  - **Uso del cluster CIMAT** permitió entrenamiento distribuido eficiente tanto para Mistral como BETO.
  
  ------
  
  ## Referencias
  
  - Jurafsky & Martin, *Speech and Language Processing* (3ª ed.)
  - Hugging Face Transformers
  - PyTorch Documentation
  - Danqi Chen et al., *BERT for Spanish (BETO)*
  - Experimentos propios sobre corpus de *Twenty One Pilots*
  
  ## Autor
  
  * César M. Aguirre Calzadilla
  
  
  
  ## Licencia
  
  Este proyecto está bajo la Licencia MIT.

