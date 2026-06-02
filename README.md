# IA_TP3_2026
# TP3 Inteligencia artificial siglo 21 2026

Este prototipo implementa un pequeño modelo de red de Hopfield para detectar y limpiar una figura en una imagen de 10x10 píxeles.

## Archivos

- `hopfield_tp3.py`: script principal del prototipo.
- `resultado_tp3.txt`: salida de ejemplo generada por la ejecución del script.

## Qué hace

- Define dos patrones de prueba de 10x10 con un aro y un bloque de referencia fijo.
- Entrena redes Hopfield con dos variantes:
  - Regla de aprendizaje de Hebb.
  - Matriz pseudoinversa.
- Aplica ruido a los patrones de prueba.
- Recupera la imagen con la red.
- Calcula el centro del aro y genera coordenadas relativas X e Y.

## Ejecución

Desde el directorio `codigo`:

```bash
python3 hopfield_tp3.py
```

Se genera también el archivo `resultado_tp3.txt` con la salida completa.
