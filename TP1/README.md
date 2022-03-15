# Descripcion
Para el problema del Cubo de Rubik 2x2x2 se implementaron los siguientes métodos de búsqueda: 
* No informados:
    - Primero en Ancho
    - Primero en Profundidad 
    - Primero en Profundidad Variable
* Informados:
    - Heurística local
    - Heurística global
    - Método A*

# Configuración
Mediante el uso de un archivo de configuración (config.json) se pueden indicar los distintos par ́ametros del problema (estado inicial, métodos utilizado y Heurística elegida)

``` json
{
"state": "0112226141317051",
"method": "global_heuristic_cost",      
"heuristic": "move_count_combination"
}
```
## Formato del estado inicial
El estado inicial es indicado mediante una cadena de 16 números, los cuales son separados en pares indicando 'id de pieza - orientacion'

## Métodos permitidos
Los valores permitidos para el campo 'method' en el archivo de configuración son:
* Métodos no informados:
-   Búsqueda primero en profundidad: "bpp"
-   Búsqueda primero a lo ancho: "bpa"
-   Búsqueda primero en profundidad variable: "bppv"

* Métodos informados:
-   Heurística local: "local_heuristic"
-   Heurística global: "global_heuristic"
-   Método A*: "global_heuristic_cost"

## Heurísticas permitidas
Los valores permitidos para el campo 'heuristic' en el archivo de configuración son:
-   sticker_groups: Determina la heurística en función de la cantidad de colores en cada cara
-   move_count_combination: Determina la heurística en función del maximo entre la cantidad de movimientos necesarios para ordenar todas las piezas y la cantidad de movimientos necesarios para orientar cada una de ellas
-   manhattan_distance: Determina la heuristica en funcion del maximo entre las distancias manhattan de cada una de las piezas y su ubicacion de destino (fijando la pieza 0 en su posicion).


# Ejecución
El programa se ejecuta mediante el comando:
```
python3.10 main.py
```
Notese que el archivo config.json posee un nombre y ubicación estaticos
