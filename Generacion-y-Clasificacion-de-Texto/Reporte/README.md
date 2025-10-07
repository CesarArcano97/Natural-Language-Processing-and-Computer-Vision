# Tarea #1 — Análisis Multimodal para MIR 🎵🤖

Este repositorio contiene el desarrollo de la **Tarea 1** de la materia *Temas Selectos de Deep Learning: Análisis Multimodal para MIR*, de la **Maestría en Cómputo Estadístico (CIMAT)**.  

## Contenido

- **`exploratorio-mir-tarea01.ipynb`** → Notebook de análisis exploratorio (EDA).  
  - Estadísticas básicas de los datos.  
  - Reducción de dimensionalidad (PCA, etc.).  
  - Visualización de la variable de respuesta (géneros).  

- **`MLP-MIR-Smooth.ipynb`** → Notebook principal de modelado.  
  - Preprocesamiento de datos:
    - Manejo de nulos.  
    - Codificación de etiquetas (`LabelEncoder`).  
    - Estandarización de variables numéricas.  
    - Embeddings para la variable categórica `artist.name`.  
  - Arquitectura de red neuronal en **PyTorch** (MLP con embeddings).  
  - Regularización (Dropout, L1, L2).  
  - Funciones de pérdida:
    - Cross-Entropy con pesos de clase.  
    - Cross-Entropy con *label smoothing*.  
    - Focal Loss (opcional).  
  - Entrenamiento y validación (con scheduler y métricas).  
  - Visualización de resultados:
    - Curvas de pérdida y F1.  
    - Matriz de confusión.  
    - Curvas ROC multiclase.  

- **`main.pdf`** → Documento de entrega de la tarea.  
  Incluye los ejercicios teóricos y prácticos:  
  1. Broadcasting en NumPy.  
  2. Redes neuronales multicapa (comparación de arquitecturas).  
  3. Regularización \(L_1\).  
  4. Backpropagation paso a paso.  
  5. Clasificador de géneros musicales (proyecto práctico con FMA).  

## Corpus

Se utiliza un subconjunto del **Free Music Archive (FMA)**, que contiene:  
- Conjunto de **entrenamiento (~15k registros)**.  
- Conjunto de **validación**.  
- Conjunto de **prueba** (sin etiquetas de género).  
- Aproximadamente **500 características numéricas** (MFCC, Chroma, Tonnetz, Spectral, etc.) más metadatos (`artist.name`, `track.genre1`).  

Referencia: [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma).

## Autor

- César M. Aguirre Calzadilla
- Maestría en Cómputo Estadístico — CIMAT
