# Configuración

Mediante el uso de un archivo de configuración (input.json) se pueden indicar los distintos parámetros del problema:

```json
{
  "max_generations": 5000,
  "max_time": null,
  "max_generations_without_improvement": null,
  "algorythm": {
    "population_count": 500,
    "mutation": {
      "probability": 0.005
    },
    "crossover": {
      "type": "OnePointCrossover",
      "params": {}
    },
    "selection": {
      "type": "BoltzmannSelection",
      "params": { "replace": false, "k": 0.001 }
    }
  }
}
```

Dentro del campo 'params' se deberan indicar los distintos parametros que necesita cada metodo.

# Métodos de cruza permitidos

Los valores permitidos para el campo 'type' en crossover son:

- OnePointCrossover -> `{ "type": "OnePointCrossover", "params": {} }`
- NPointCrossover -> `{ "type": "NPointCrossover", "params": {"points": int} }`
- UniformCrossover -> `{ "type": "UniformCrossover", "params": {} }`

# Métodos de seleccion permitidos

Los valores permitidos para el campo 'type' en selection sson:

- EliteSelection -> `{ "type": "EliteSelection", "params": {} }`
- RouletteSelection -> `{ "type": "RouletteSelection", "params": {"replace": true | false} }`
- RankSelection -> `{ "type": "RankSelection", "params": {"replace": true | false} }`
- TournamentSelection -> `{ "type": "TournamentSelection", "params": {"replace": true | false, "threshold": float} }`
- BoltzmannSelection -> `{ "type": "BoltzmannSelection", "params": {"replace": true | false, "k": float, "T0": float = 10000, "Tc": float = 15} }`
- TruncatedSelection -> `{ "type": "TruncatedSelection", "params": {"truncate_count": int} }`

# Ejecución

Para poder ejecutar el programa, se requiere una version de python superior a la 3.10.

El programa se ejecuta mediante el comando:

```
python3.10 main.py
```
