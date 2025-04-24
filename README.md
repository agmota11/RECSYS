# Proyecto de Sistemas de Recomendación (Práctica 2)

## Estructura del Notebook

### Train data all_more

| Categoría                  | Notebooks                                                                                   |
|---------------------------|---------------------------------------------------------------------------------------------|
| Preprocesamiento de texto | - Procesando texto de manera clásica (`TF-IDF`, `TruncatedSVD`) <br> - Procesando texto DL (`mpnet-base-v2`) |
| ML           | - `RandomForestReg` <br> - `KNeighborsRegressor` <br> - `GradientBoostingRegressor` <br> - `XGBRegressor` <br> - `LGBMRegressor` |
| Modelos de deep learning  | - `NeuralNetwork` <br> - `Transformer` <br> - `Transformer + regularization` |
| Métodos ensemble           | - `VotingRegressor ML` <br> - `Stacking ML` <br> - `Ensemble`                                          |

### Hiperparámetros

Pruebas específicas de optimización y reducción dimensional (`Optuna`/`Ray Tune`):

- Reducción dimensional (TruncatedSVD y PCA)
- Tuning de XGBoost
- Tuning de redes neuronales

### Test models

Evaluación de modelos:

- Resultados guardados en archivos CSV

## Herramientas Auxiliares

### Docker Ray Tune

Contiene configuraciones para pruebas en diferentes entornos:

- `docker-compose.yaml` y `Dockerfile`
- Solo para pruebas aisladas o busqueda de hiperparámetros.

---

## Otras ideas/librerias

| Herramienta / Enfoque         | Descripción |
|------------------------------|-------------|
| **RecBole**                  | [Recbole.io](https://recbole.io/model_list.html) |
| **NVIDIA Merlin**            | [NVIDIA Merlin Recommender Workflows](https://developer.nvidia.com/merlin) |
| **TensorFlow Recommenders** | [TensorFlow Recommenders](https://www.tensorflow.org/recommenders?hl=es-419) |
| **Cornac** | [Cornac](https://github.com/PreferredAI/cornac) |
| **TensorFlow Recommenders** | [TensorFlow Recommenders](https://www.tensorflow.org/recommenders?hl=es-419) |
| **Awesome** | [Awesome_0](https://github.com/loserChen/Awesome-Recommender-System) [Awesome_1](https://github.com/USTC-StarTeam/Awesome-Large-Recommendation-Models) [Awesome_2](https://github.com/creyesp/Awesome-recsys) |
| **Embeddings + CNN-LSTM**  | [Rafay et al., 2020 (IEEE)](https://ieeexplore.ieee.org/document/9283501). |

---

