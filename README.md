# 🎵 Análise de Popularidade de Músicas no Spotify

## Descrição

Projeto de Machine Learning para prever popularidade de músicas e classificar gêneros musicais usando características de áudio do Spotify.

## Objetivos

- **Regressão:** Prever popularidade (0-100) usando features de áudio
- **Classificação:** Identificar macro-gêneros musicais (9 classes)

## Dataset

- **Fonte:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Licença:** CC0 (Domínio Público)
- **Tamanho:** ~114.000 músicas, 114 gêneros
- **Features:** 21 colunas incluindo características de áudio (danceability, energy, loudness, etc.)

---

## 📊 Resultados e Descobertas

### Análise Exploratória (EDA)

#### Correlações com Popularidade
| Feature | Correlação |
|---------|------------|
| loudness | +0.047 |
| danceability | +0.034 |
| tempo | +0.012 |
| instrumentalness | -0.092 |
| speechiness | -0.045 |
| valence | -0.043 |

**Insight Principal:** Nenhuma feature de áudio tem correlação forte com popularidade. Isso sugere que o sucesso de uma música depende mais de fatores externos (artista, marketing, timing) do que características técnicas.

#### Feature Engineering
A única feature criada com correlação significativa foi o **target encoding por gênero** (`genre_popularity`), que obteve correlação de **0.503** com a popularidade.

---

### 📈 Testes Estatísticos

#### T-Test: Conteúdo Explícito vs Popularidade
- **Hipótese Nula (H₀):** Músicas explícitas têm a mesma popularidade
- **T-statistic:** 14.898
- **P-value:** 0.000

**Resultado:** ✅ Rejeitamos H₀. Músicas com conteúdo explícito são **significativamente mais populares**.

| Tipo | Média de Popularidade |
|------|----------------------|
| Explícitas | 36.45 |
| Não Explícitas | 32.94 |

#### ANOVA: Popularidade entre Gêneros
- **F-statistic:** 180.033
- **P-value:** ≈ 0

**Resultado:** ✅ Existem diferenças significativas de popularidade entre gêneros.

**Ranking de Popularidade por Macro-Gênero:**
| Posição | Gênero | Média |
|---------|--------|-------|
| 1º | Rock | 36.70 |
| 2º | Pop | 35.72 |
| 3º | Latin-World | 33.52 |
| 4º | Classical | 32.63 |
| 5º | Hip-Hop | 32.60 |
| 6º | Electronic | 32.30 |
| 7º | Metal | 30.70 |
| 8º | Country-Folk | 29.35 |
| 9º | Jazz-Blues | 27.30 |

#### Chi-Quadrado: Gênero vs Conteúdo Explícito
- **Chi²:** 2884.421
- **P-value:** ≈ 0

**Resultado:** ✅ Existe forte associação entre gênero musical e conteúdo explícito.

| Gênero | % Explícitas |
|--------|--------------|
| Hip-Hop | 18.4% |
| Metal | 18.4% |
| Pop | 10.4% |
| Rock | 7.7% |
| Classical | 1.0% |

---

### 🤖 Modelos de Machine Learning

#### Regressão (Prever Popularidade 0-100)

| Modelo | MAE | RMSE | R² | Melhoria vs Baseline |
|--------|-----|------|-----|----------------------|
| Baseline (média) | 18.87 | 22.28 | 0.00 | - |
| Linear Simples | 14.11 | 19.26 | 0.25 | +25% |
| Linear Múltipla | 14.08 | 19.23 | 0.26 | +26% |
| Polinomial (grau 2) | 14.10 | 19.22 | 0.26 | +26% |
| **Random Forest** | **11.46** | **16.73** | **0.44** | **+71%** |

**Melhor Modelo:** Random Forest Regressor

**Interpretação das Métricas:**
- **MAE = 11.46:** Em média, o modelo erra ~11 pontos na escala de 0-100
- **R² = 0.44:** O modelo explica 44% da variância da popularidade
- **Melhoria:** 71% melhor que simplesmente prever a média

#### Classificação (Prever Macro-Gênero)

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Baseline | 20.32% | - | - | - | - |
| Naive Bayes | 33.87% | 36.12% | 33.87% | 30.88% | 0.73 |
| Regressão Logística | 40.30% | 37.61% | 40.30% | 37.09% | 0.76 |
| Gradient Boosting | 48.95% | - | - | - | - |
| **Random Forest (otimizado)** | **54.59%** | - | - | - | - |

**Melhor Modelo:** Random Forest Classifier (tunado via RandomizedSearchCV)

**Hiperparâmetros Otimizados:**
```python
{
    'n_estimators': 100,
    'max_depth': 25,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'log2',
    'class_weight': None
}
```

---

### 💡 Conclusões

1. **Features de áudio sozinhas não determinam popularidade**
   - A maior correlação encontrada foi apenas 0.092 (instrumentalness)
   - O gênero musical (target encoding) foi a feature mais preditiva

2. **Modelos não-lineares superam significativamente os lineares**
   - Random Forest obteve R² = 0.44 vs 0.26 dos modelos lineares
   - Gradient Boosting e Random Forest dominaram na classificação

3. **Músicas explícitas são mais populares**
   - Diferença estatisticamente significativa (p < 0.001)
   - Média 3.5 pontos maior que músicas não-explícitas

4. **Rock e Pop lideram em popularidade**
   - Gêneros mais tradicionais têm médias mais altas
   - Hip-Hop e Metal têm maior % de conteúdo explícito

5. **Limitações do modelo**
   - Não consegue prever popularidades extremas (> 60)
   - Classificação de gêneros é limitada (~55% accuracy) devido à sobreposição entre gêneros

---

### 🚀 Trabalhos Futuros

- Incluir features de artista (seguidores, histórico de lançamentos)
- Análise de séries temporais (tendências de popularidade)
- Adicionar features de letras usando NLP
- Testar modelos de Deep Learning
- Incluir dados de playlists e contexto de escuta

---

## Estrutura do Repositório

```
├── src/
│   └── notebook.ipynb          # Notebook principal com toda análise
├── models/
│   ├── best_reg_model.pkl      # Modelo de regressão salvo
│   └── results_reg_comparison.csv
├── dataset.csv                 # Dataset original
├── dataset_with_features.csv   # Dataset com feature engineering
├── requirements.txt            # Dependências
├── README.md                   # Este arquivo
└── LICENSE                     # Licença MIT
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
jupyter notebook src/notebook.ipynb
```

## Tecnologias

- Python 3.10+
- pandas, numpy, seaborn, matplotlib
- scikit-learn, statsmodels, scipy
- PyCaret (AutoML)

## Autor

Gabriel Santos - CESUPA

## Licença

MIT License
