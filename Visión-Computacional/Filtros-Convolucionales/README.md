# Filtros de Imagen en CPU vs. GPU con CUDA

![Texto alternativo para la imagen](Natural-Language-Processing-and-Computer-Vision/Visión-Computacional/Filtros-Convolucionales/imagenes/Filtros_GPU/La_Ofrende.png)

Este proyecto implementa y compara el rendimiento de tres filtros clásicos de procesamiento de imágenes bajo dos enfoques de ejecución:

1. **Ejecución en CPU**: Utilizando librerías estándar de Python como `NumPy` y `SciPy`.
2. **Ejecución en GPU**: Acelerando los cálculos con `CuPy` y kernels de CUDA C++ personalizados.

El objetivo principal es cuantificar la ganancia de velocidad (_speedup_) que se obtiene al paralelizar estas operaciones en una GPU, demostrando su eficacia para tareas comunes en Visión Computacional.

## Descripción del Proyecto

El código está estructurado en tres cuadernos de Jupyter:

1. `Filtros_CPU.ipynb`: Contiene la implementación y pruebas de los filtros ejecutados exclusivamente en la CPU.
2. `Filtros_GPU.ipynb`: Implementa los mismos filtros utilizando `cupy.RawKernel` para definir y ejecutar kernels de CUDA personalizados en la GPU.
3. `Filtros_Comparación.ipynb`: Realiza una comparativa directa de los tiempos de ejecución entre ambas implementaciones y visualiza los resultados, incluyendo el _speedup_.

Los experimentos se realizaron sobre diversas obras de arte, como "La Ofrenda" (1913) de Saturnino Herrán, para analizar el impacto de los filtros en imágenes con texturas complejas y variaciones de color.

### Filtros Implementados

Se implementaron los siguientes tres filtros fundamentales:

- **Filtro de Media**: Un operador de suavizado que reduce el ruido y las variaciones de intensidad promediando los valores de los píxeles en una vecindad definida por un kernel. El tamaño del kernel es configurable.
- **Magnitud del Gradiente**: Un método de detección de bordes que aproxima el gradiente de la imagen mediante diferencias finitas en los ejes X y Y. Cuantifica la tasa de cambio de intensidad, resaltando discontinuidades.
- **Filtro Laplaciano**: Un detector de bordes de segundo orden, sensible a las discontinuidades finas y capaz de identificar si un píxel pertenece a un lado claro u oscuro de un borde.

## Resultados

Los resultados demuestran una aceleración (_speedup_) significativa a favor de la implementación en GPU, especialmente en operaciones computacionalmente intensivas.

- Para el **Filtro de Media** con un kernel de 20x20, la GPU fue aproximadamente **30 veces más rápida**.
- Para el **Filtro de Gradiente**, la GPU mostró un rendimiento **6 veces superior**.
- Para el **Filtro Laplaciano**, la aceleración fue de aproximadamente **2 veces**.

La diferencia en el _speedup_ se atribuye a la carga computacional de cada filtro. El filtro de media con un kernel grande requiere una cantidad masiva de operaciones repetitivas por cada píxel, un escenario ideal para la arquitectura masivamente paralela de las GPUs.

## Cómo Utilizar este Repositorio

### Requisitos

- Python 3.x
- NumPy
- CuPy (con una instalación de CUDA compatible)
- Matplotlib
- OpenCV (`cv2`)
- Jupyter Notebook o JupyterLab

### Ejecución

1. Clona este repositorio en tu máquina local.
2. Asegúrate de tener todos los requisitos instalados.
3. Abre y ejecuta los cuadernos de Jupyter en el siguiente orden para replicar los resultados:
    - `Filtros_CPU.ipynb`
    - `Filtros_GPU.ipynb`
    - `Filtros_Comparación.ipynb`

Puedes modificar la variable `FILENAME` en los cuadernos para probar los filtros con tus propias imágenes.

### Autor 
* César M. Aguirre Calzadilla

### Licencia 

Este proyecto está bajo la Licencia MIT.
