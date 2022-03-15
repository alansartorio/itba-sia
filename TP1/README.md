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
Mediante el uso de un archivo de configuración (config.json) se pueden indicar los distintos parámetros del problema (estado inicial, métodos utilizado y Heurística elegida)

``` json
{
    "state": "0112226141317051",
    "method": "global_heuristic_cost",      
    "heuristic": "move_count_combination"
}
```

En el caso en el que el metodo de busqueda deseado no requiera de una función heurística, se debera o bien omitir el campo "heuristic" del config.json o asignarle un string vacío.

```json
{
    "state": "2071411031025160",
    "method": "bpa"
}
```

## Formato del estado inicial
El estado inicial es indicado mediante una cadena de 16 números, los cuales son separados en pares indicando 'id de pieza - orientacion'

## Métodos permitidos
Los valores permitidos para el campo 'method' en el archivo de configuración son:
* Métodos no informados:
    - Búsqueda primero en profundidad: "bpp"
    - Búsqueda primero a lo ancho: "bpa"
    - Búsqueda primero en profundidad variable: "bppv"

* Métodos informados:
    - Heurística local: "local_heuristic"
    - Heurística global: "global_heuristic"
    - Método A*: "global_heuristic_cost"

## Heurísticas permitidas
Los valores permitidos para el campo 'heuristic' en el archivo de configuración son:
-   sticker_groups: Determina la heurística en función de la cantidad de colores en cada cara
-   move_count_combination: Determina la heurística en función del maximo entre la cantidad de movimientos necesarios para ordenar todas las piezas y la cantidad de movimientos necesarios para orientar cada una de ellas
-   manhattan_distance: Determina la heuristica en funcion del maximo entre las distancias manhattan de cada una de las piezas y su ubicacion de destino (fijando la pieza 0 en su posicion).

# Archivos de configuracion
Se poseen dos ejemplos de configuracion (config.example1.json y config.example2.json) para poder ejecutarlos, estos deberan ser renombrados con el nombre 'config.json'

# Ejecución
Para poder ejecutar el programa, se requiere una version de python superior a la 3.10 y tener instalada la libreria "typing_extensions" (esta libreria no le provee funcionalidad a nuestro programa, solo nos permite tener un mejor tipado durante el desarrollo).
Esta libreria se puede instalar mediante: `python3.10 -m pip install typing_extensions`.
El programa se ejecuta mediante el comando:
```
python3.10 main.py
```
Notese que el archivo config.json posee un nombre y ubicación estaticos

# Funcionamiento Basico
Al ejecutar el programa `main.py`, este leerá el archivo de entrada `config.json` y comenzará a realizar la busqueda. Cuando éste termine, escribirá un archivo llamado `solution.json` de la forma:
```json
{
  "result": "success",
  "processing_time": 0.0835421085357666,
  "search_config": {
    "heuristic": "move_count_combination",
    "method": "global_heuristic_cost",
    "state": "1122304250610271"
  },
  "solution_depth": 9,
  "solution_cost": 8,
  "expanded_nodes": 946,
  "border_nodes": 739,
  "solution": {
    "initial_state": "1122304250610271",
    "intermediate_states": [
      "2141123050610271",
      "1241023020615171",
      "0111304220615171",
      "3111504200612171",
      "0231504261112171",
      "6031024220115171",
      "2262024211315171"
    ],
    "final_state": "0121416111315171"
  }
}
```

Además de esto, el programa imprimirá por pantalla de la terminal los movimientos a hacer para llegar a la solución encontrada y el estado del cubo despues de cada paso (en color).

Por último, el programa escribirá un segundo archivo llamado `visualization.txt` que contiene el estado inicial (mezclado) y los pasos necesarios para resolverlo. Fué pensado para la visualizacion en 3D con [este programa](https://github.com/alansartorio/rubik) (escrito por nosotros). 

Este archivo tiene la siguiente forma:

```
fl
ur

rd
bf

uf
bd

rb
df

lb
ur

ud
ll

===
U
F
U'
F
R
F'
R
U'
```
