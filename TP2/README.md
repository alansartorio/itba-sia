# Configuración
Mediante el uso de un archivo de configuración (input.json) se pueden indicar los distintos parámetros del problema:

``` json
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
			"params": {"replace": false, "k": 0.001}
		}
	}
}

```
Dentro del campo 'params' se deberan indicar los distintos parametros que necesita cada metodo.

# Métodos de cruza permitidos
Los valores permitidos para el campo 'type' en crossover son:
  - OnePointCrossover
  - NPointCrossover
  - UniformCrossover

# Métodos de seleccion permitidos
Los valores permitidos para el campo 'type' en selection sson:
  - EliteSelection
  - RankSelection
  - RouletteSelection',
  - TournamentSelection',
  - BoltzmannSelection',
  - TruncatedSelection',
  
# Ejecución
Para poder ejecutar el programa, se requiere una version de python superior a la 3.10.

El programa se ejecuta mediante el comando:
```
python3.10 main.py
```
