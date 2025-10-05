Estructura del Proyecto
=======================
La estructura del proyecto está organizada de la siguiente manera:

```bash
PartB/
│
├── data/
│   ├── raw/
│   │   └── MeIA_2025_train.csv
│   └── processed/
│       └── classif/
│           └── five/
│               └── (aquí se podrán guardar features o vocab.pkl)
│
├── src/
│   ├── data/
│   │   └── datasets.py        # Implementa lectura, limpieza, tokenización y KFold
│   ├── train/
│   │   ├── train_rnnclf.py
│   │   └── train_cnnclf.py
│   ├── models/
│   │   ├── rnn_text.py
│   │   └── cnn_text.py
│   └── eval/
│       └── eval_metrics.py
│
└── results/
    ├── logs/
    └── reports/
```
