"""
Spotify Genre Classifier - Modal Deploy
Classificador de gênero musical usando Random Forest

Autor: Gabriel Santos - CESUPA
"""

import modal

# Criar a app Modal
app = modal.App("spotify-genre-classifier")

# Imagem com todas as dependências
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas==2.1.4",
    "numpy==1.26.4",
    "scikit-learn==1.4.2",
    "joblib",
)

# Volume para persistir o modelo treinado
volume = modal.Volume.from_name("spotify-model-volume", create_if_missing=True)
MODEL_DIR = "/models"


MACRO_GENRE_MAP = {
    # Rock
    'rock': 'rock', 'alt-rock': 'rock', 'hard-rock': 'rock', 'psych-rock': 'rock',
    'punk-rock': 'rock', 'grunge': 'rock', 'punk': 'rock', 'emo': 'rock',
    'indie': 'rock', 'indie-pop': 'rock', 'garage': 'rock', 'rockabilly': 'rock',
    'rock-n-roll': 'rock', 'british': 'rock', 'j-rock': 'rock', 'goth': 'rock',

    # Pop
    'pop': 'pop', 'power-pop': 'pop', 'synth-pop': 'pop', 'k-pop': 'pop',
    'j-pop': 'pop', 'cantopop': 'pop', 'mandopop': 'pop', 'dance': 'pop',
    'disco': 'pop', 'pop-film': 'pop', 'swedish': 'pop',

    # Electronic
    'electronic': 'electronic', 'edm': 'electronic', 'house': 'electronic',
    'deep-house': 'electronic', 'progressive-house': 'electronic', 'chicago-house': 'electronic',
    'techno': 'electronic', 'detroit-techno': 'electronic', 'minimal-techno': 'electronic',
    'trance': 'electronic', 'dubstep': 'electronic', 'drum-and-bass': 'electronic',
    'breakbeat': 'electronic', 'electro': 'electronic', 'idm': 'electronic',
    'ambient': 'electronic', 'chill': 'electronic', 'trip-hop': 'electronic',
    'dub': 'electronic', 'garage': 'electronic', 'hardstyle': 'electronic',
    'j-dance': 'electronic', 'j-idol': 'electronic',

    # Hip-Hop/R&B
    'hip-hop': 'hip-hop', 'r-n-b': 'hip-hop', 'reggaeton': 'hip-hop',
    'dancehall': 'hip-hop',

    # Metal
    'metal': 'metal', 'heavy-metal': 'metal', 'black-metal': 'metal',
    'death-metal': 'metal', 'metalcore': 'metal', 'grindcore': 'metal',
    'hardcore': 'metal', 'industrial': 'metal',

    # Classical/Instrumental
    'classical': 'classical', 'piano': 'classical', 'opera': 'classical',
    'new-age': 'classical', 'sleep': 'classical', 'study': 'classical',
    'ambient': 'classical', 'guitar': 'classical', 'acoustic': 'classical',
    'songwriter': 'classical', 'singer-songwriter': 'classical',

    # Jazz/Blues/Soul
    'jazz': 'jazz-blues', 'blues': 'jazz-blues', 'soul': 'jazz-blues',
    'funk': 'jazz-blues', 'groove': 'jazz-blues',

    # Country/Folk
    'country': 'country-folk', 'folk': 'country-folk', 'bluegrass': 'country-folk',
    'honky-tonk': 'country-folk', 'americana': 'country-folk',

    # Latin/World
    'latin': 'latin-world', 'latino': 'latin-world', 'salsa': 'latin-world',
    'samba': 'latin-world', 'bossa-nova': 'latin-world', 'brazilian': 'latin-world',
    'brazil': 'latin-world', 'mpb': 'latin-world', 'pagode': 'latin-world',
    'sertanejo': 'latin-world', 'forro': 'latin-world', 'tango': 'latin-world',
    'reggae': 'latin-world', 'ska': 'latin-world', 'afrobeat': 'latin-world',
    'world-music': 'latin-world', 'indian': 'latin-world', 'malay': 'latin-world',
    'turkish': 'latin-world', 'iranian': 'latin-world', 'spanish': 'latin-world',
    'french': 'latin-world', 'german': 'latin-world',
}


@app.function(
    image=image,
    gpu="T4",
    volumes={MODEL_DIR: volume},
    timeout=1000,
)
def train_model():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib
    import os

    print("Carregando dataset do Spotify...")

    try:
        df = pd.read_csv("/models/dataset.csv")
        print(f"Dataset carregado: {len(df)} músicas")
    except FileNotFoundError:
        # disparar erro
        raise FileNotFoundError("Dataset não encontrado")

    print("\nMapeando gêneros para macro-gêneros...")
    df['macro_genre'] = df['track_genre'].map(MACRO_GENRE_MAP).fillna('other')

    df = df[df['macro_genre'] != 'other']
    print(f"Músicas após mapeamento: {len(df)}")

    # Distribuição dos macro-gêneros
    print("\nDistribuição dos macro-gêneros:")
    for genre, count in df['macro_genre'].value_counts().items():
        print(f"   {genre}: {count}")

    # Features para classificação
    features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'valence', 'tempo']

    X = df[features].values
    y = df['macro_genre'].values

    # Encoder para gêneros
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scaler para normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\n📊 Classes encontradas: {label_encoder.classes_.tolist()}")
    print(f"📊 Amostras de treino: {len(X_train)}")
    print(f"📊 Amostras de teste: {len(X_test)}")

    print("\n🚀 Treinando Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_classifier.fit(X_train, y_train)

    # Métricas
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(rf_classifier, X_scaled, y_encoded, cv=5)

    print(f"\n📈 MÉTRICAS DO MODELO:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Feature importance
    feature_importance = dict(zip(features, rf_classifier.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n🎯 IMPORTÂNCIA DAS FEATURES:")
    for feat, imp in sorted_importance:
        print(f"   {feat}: {imp:.4f}")

    # Classification report
    print(f"\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Salvar modelos
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(rf_classifier, f"{MODEL_DIR}/rf_classifier.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.joblib")

    # Salvar metadados
    metadata = {
        'features': features,
        'classes': label_encoder.classes_.tolist(),
        'metrics': {
            'accuracy': float(accuracy),
            'cv_score_mean': float(cv_scores.mean()),
            'cv_score_std': float(cv_scores.std()),
        },
        'feature_importance': {k: float(v) for k, v in feature_importance.items()}
    }
    joblib.dump(metadata, f"{MODEL_DIR}/metadata.joblib")

    volume.commit()

    print("\n✅ Modelo salvo com sucesso!")
    return metadata


@app.function(
    image=image,
    gpu="T4",
    volumes={MODEL_DIR: volume},
)
def predict_genre(features: dict) -> dict:
    import joblib
    import numpy as np

    # Carregar modelos
    model = joblib.load(f"{MODEL_DIR}/rf_classifier.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")
    metadata = joblib.load(f"{MODEL_DIR}/metadata.joblib")

    # Preparar features
    feature_names = metadata['features']
    X = np.array([[features.get(f, 0) for f in feature_names]])
    X_scaled = scaler.transform(X)

    # Predizer
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    # Decodificar
    genre = label_encoder.inverse_transform([prediction])[0]

    # Todos os gêneros com probabilidades
    all_genres = [
        {
            "genre": label_encoder.inverse_transform([i])[0],
            "probability": round(float(probabilities[i]) * 100, 2)
        }
        for i in np.argsort(probabilities)[::-1]
    ]

    return {
        "predicted_genre": genre,
        "confidence": round(float(max(probabilities)) * 100, 2),
        "all_genres": all_genres
    }


@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
)
def get_model_info() -> dict:
    """Retorna informações sobre o modelo treinado."""
    import joblib

    try:
        metadata = joblib.load(f"{MODEL_DIR}/metadata.joblib")
        return {
            "status": "trained",
            "model_type": "Random Forest Classifier",
            "classes": metadata['classes'],
            "features": metadata['features'],
            "metrics": metadata['metrics'],
            "feature_importance": metadata['feature_importance']
        }
    except FileNotFoundError:
        return {"status": "not_trained", "message": "Execute train_model() primeiro"}


# API Web com FastAPI
@app.function(
    image=image.pip_install("fastapi[standard]"),
    gpu="T4",
    volumes={MODEL_DIR: volume},
)
@modal.asgi_app()
def web_app():
    """API REST para classificação de gênero musical."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import joblib
    import numpy as np

    api = FastAPI(
        title="Spotify Genre Classifier",
        description="API para classificação de gênero musical baseada em features de áudio do Spotify",
        version="1.0.0"
    )

    class MusicFeatures(BaseModel):
        danceability: float = Field(0.5, ge=0, le=1, description="Quão dançante é a música (0-1)")
        energy: float = Field(0.5, ge=0, le=1, description="Intensidade e atividade (0-1)")
        loudness: float = Field(-10.0, ge=-60, le=0, description="Volume em dB (-60 a 0)")
        speechiness: float = Field(0.1, ge=0, le=1, description="Presença de palavras faladas (0-1)")
        acousticness: float = Field(0.3, ge=0, le=1, description="Quão acústica é a música (0-1)")
        instrumentalness: float = Field(0.0, ge=0, le=1, description="Ausência de vocais (0-1)")
        valence: float = Field(0.5, ge=0, le=1, description="Positividade musical (0-1)")
        tempo: float = Field(120.0, ge=0, le=300, description="Batidas por minuto (BPM)")

    @api.get("/")
    def root():
        return {
            "message": "Spotify Genre Classifier API",
            "docs": "/docs",
            "endpoints": {
                "POST /predict": "Classificar gênero de uma música",
                "GET /model-info": "Informações do modelo",
                "GET /health": "Health check"
            }
        }

    @api.get("/health")
    def health():
        return {"status": "healthy"}

    @api.get("/model-info")
    def model_info():
        try:
            metadata = joblib.load(f"{MODEL_DIR}/metadata.joblib")
            return {
                "status": "trained",
                "model_type": "Random Forest Classifier",
                "classes": metadata['classes'],
                "accuracy": f"{metadata['metrics']['accuracy']*100:.2f}%",
                "features": metadata['features'],
                "feature_importance": metadata['feature_importance']
            }
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Modelo não treinado")

    @api.post("/predict")
    def predict(features: MusicFeatures):
        """Classifica o gênero musical baseado nas features de áudio."""
        try:
            model = joblib.load(f"{MODEL_DIR}/rf_classifier.joblib")
            scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
            label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")
            metadata = joblib.load(f"{MODEL_DIR}/metadata.joblib")

            feature_names = metadata['features']
            X = np.array([[getattr(features, f) for f in feature_names]])
            X_scaled = scaler.transform(X)

            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]

            genre = label_encoder.inverse_transform([prediction])[0]

            # Top 3 gêneros
            top_indices = np.argsort(probabilities)[::-1][:3]
            top_genres = [
                {
                    "genre": label_encoder.inverse_transform([i])[0],
                    "probability": round(float(probabilities[i]) * 100, 2)
                }
                for i in top_indices
            ]

            return {
                "predicted_genre": genre,
                "confidence": round(float(max(probabilities)) * 100, 2),
                "top_3_genres": top_genres
            }
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Modelo não treinado. Execute o treinamento primeiro.")

    return api


# Entrypoint local para testes
@app.local_entrypoint()
def main():
    print("🎵 Spotify Genre Classifier - Modal Deploy")
    print("=" * 50)

    # Treinar modelo
    print("\n📚 Treinando classificador de gênero...")
    result = train_model.remote()
    print(f"\nMetadados: {result}")

    # Testar predição
    print("\n🔮 Testando predição de gênero...")

    # Exemplo 1: Música eletrônica/dançante
    test_electronic = {
        'danceability': 0.85,
        'energy': 0.9,
        'loudness': -5.0,
        'speechiness': 0.05,
        'acousticness': 0.1,
        'instrumentalness': 0.7,
        'valence': 0.8,
        'tempo': 128.0
    }
    print("\n🎹 Teste 1 - Música eletrônica/dançante:")
    result1 = predict_genre.remote(test_electronic)
    print(f"   Gênero: {result1['predicted_genre']} ({result1['confidence']}%)")

    # Exemplo 2: Música clássica/acústica
    test_classical = {
        'danceability': 0.2,
        'energy': 0.3,
        'loudness': -20.0,
        'speechiness': 0.03,
        'acousticness': 0.95,
        'instrumentalness': 0.9,
        'valence': 0.4,
        'tempo': 80.0
    }
    print("\n🎻 Teste 2 - Música clássica/acústica:")
    result2 = predict_genre.remote(test_classical)
    print(f"   Gênero: {result2['predicted_genre']} ({result2['confidence']}%)")

    # Exemplo 3: Hip-hop
    test_hiphop = {
        'danceability': 0.75,
        'energy': 0.7,
        'loudness': -6.0,
        'speechiness': 0.3,
        'acousticness': 0.1,
        'instrumentalness': 0.0,
        'valence': 0.5,
        'tempo': 95.0
    }
    print("\n🎤 Teste 3 - Hip-hop:")
    result3 = predict_genre.remote(test_hiphop)
    print(f"   Gênero: {result3['predicted_genre']} ({result3['confidence']}%)")

    print("\n" + "=" * 50)
    print("✅ Treinamento concluído!")
    print("\nPróximos passos:")
    print("  - Testar API: modal serve modal_app.py")
    print("  - Deploy:     modal deploy modal_app.py")
