# Análise de Popularidade de Músicas no Spotify

## Descrição

Projeto de Machine Learning para prever popularidade de músicas e classificar gêneros musicais usando características de áudio do Spotify.

## Objetivos

- **Regressão:** Prever popularidade (0-100) usando features de áudio
- **Classificação:** Identificar macro-gêneros musicais (9 classes)

## Dataset

- **Fonte:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Licença:** CC0 (Domínio Público)
- **Tamanho:** ~114.000 músicas, 125 gêneros

## Estrutura do Repositório

```
├── src/
│   └── notebook.ipynb      # Notebook principal
├── dataset.csv            # Dados
├── requirements.txt       # Dependências
├── README.md             # Este arquivo
└── LICENSE               # Licença MIT
```

## Instalação

```bash
# Clonar repositório
git clone https://github.com/gabrielmatsan/machine-learning-spotify.git
cd machine-learning-spotify

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Execução

```bash
jupyter notebook src/projeto.ipynb
```

## Resultados Principais

### Regressão (Prever Popularidade)

| Modelo           | MAE   | RMSE  | R²            |
| ---------------- | ----- | ----- | -------------- |
| Baseline         | 18.87 | 22.28 | 0.00           |
| Linear Múltipla | 14.08 | 19.23 | 0.26           |
| Random Forest    | 11.46 | 16.73 | **0.44** |

### Classificação (Prever Gênero)

| Modelo        | Accuracy         | F1-Score | AUC-ROC |
| ------------- | ---------------- | -------- | ------- |
| Baseline      | 20.32%           | -        | -       |
| Logística    | 40.30%           | 37.09%   | 0.76    |
| Random Forest | **54.59%** | -        | -       |

## Tecnologias

- Python 3.10+
- pandas, numpy, seaborn, matplotlib
- scikit-learn, statsmodels, pycaret

## Autor

Gabriel Santos - CESUPA

## Licença

MIT License

```

### 2. requirements.txt
```

pandas>=2.0.0
numpy>=1.24.0
seaborn>=0.12.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
scipy>=1.11.0
jupyter>=1.0.0

```

### 3. LICENSE
```

MIT License

Copyright (c) 2024 Gabriel Santos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
