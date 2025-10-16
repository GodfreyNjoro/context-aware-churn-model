# Explainable Context-Aware NLP Churn Prediction System

[![CI](https://github.com/GodfreyNjoro/context-aware-churn-model/workflows/CI/badge.svg)](https://github.com/GodfreyNjoro/context-aware-churn-model/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An advanced machine learning system for predicting customer churn using Natural Language Processing (NLP) and explainable AI techniques. This project combines contextual customer interaction analysis with state-of-the-art interpretability methods to provide actionable insights for customer retention.

## 🌟 Features

- **Context-Aware Predictions**: Analyzes customer interactions, sentiment, and behavioral patterns
- **NLP Integration**: Processes customer feedback and support tickets using transformer-based embeddings
- **Explainable AI**: SHAP (SHapley Additive exPlanations) values for model interpretability
- **RESTful API**: FastAPI-based service for real-time predictions
- **Docker Support**: Containerized deployment for easy scaling
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Comprehensive Monitoring**: Built-in health checks and logging

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

The system consists of several key components:

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ User Data    │  │ Interactions │  │ Transactions │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature Engineering                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Behavioral Features  • NLP Embeddings              │   │
│  │ • Sentiment Analysis   • Interaction Patterns        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML Model Layer                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Random Forest / Gradient Boosting Classifier         │   │
│  │ + SHAP Explainer                                     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ /predict     │  │ /explain     │  │ /health      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Key Technologies

- **Machine Learning**: scikit-learn, XGBoost, Random Forest
- **NLP**: Transformers, Sentence-BERT, spaCy
- **Explainability**: SHAP, LIME
- **API**: FastAPI, Pydantic
- **Deployment**: Docker, Docker Compose
- **CI/CD**: GitHub Actions

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- Git

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/GodfreyNjoro/context-aware-churn-model.git
   cd context-aware-churn-model
   ```

2. **Run the setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This script will:
   - Create a virtual environment
   - Install all dependencies
   - Download required NLP models
   - Generate sample data
   - Train the initial model

### Manual Installation

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create directories
mkdir -p data/raw data/processed models/saved logs
```

## ⚡ Quick Start

### Using the Setup Script

```bash
./setup.sh
```

### Manual Quick Start

1. **Generate sample data**
   ```bash
   python scripts/generate_data.py
   ```

2. **Train the model**
   ```bash
   python scripts/train_model.py
   ```

3. **Start the API server**
   ```bash
   uvicorn src.api.main:app --reload
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## 📖 Usage

### Python API

```python
from src.models.churn_model import ExplainableChurnModel
import pandas as pd

# Load trained model
model = ExplainableChurnModel()
model.load_model("models/saved/churn_model.pkl")

# Prepare features
features = pd.DataFrame([{
    'age': 35,
    'tenure_days': 180,
    'monthly_spend': 75.50,
    'total_interactions': 15,
    # ... other features
}])

# Make prediction
prediction = model.model.predict(features)[0]
probability = model.model.predict_proba(features)[0][1]

# Get explanation
explanation = model.explain_prediction(features, sample_idx=0)
print(f"Churn Probability: {probability:.2%}")
print(f"Top Risk Factors: {explanation['top_features'][:3]}")
```

### REST API

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "tenure_days": 180,
    "subscription_tier": "premium",
    "monthly_spend": 75.50,
    "total_interactions": 15,
    "resolution_rate": 0.85,
    "sentiment_positive": 8,
    "sentiment_neutral": 5,
    "sentiment_negative": 2
  }'

# Get feature importance
curl http://localhost:8000/feature-importance?top_n=10
```

### Response Example

```json
{
  "churn_probability": 0.73,
  "churn_prediction": true,
  "risk_level": "high",
  "top_risk_factors": [
    {"feature": "recency_days", "shap_value": 0.15},
    {"feature": "sentiment_negative", "shap_value": 0.12},
    {"feature": "resolution_rate", "shap_value": -0.10}
  ],
  "recommendation": "High churn risk. Immediate intervention recommended with personalized offers."
}
```

## 📚 API Documentation

### Endpoints

#### `GET /`
Root endpoint with API information.

#### `GET /health`
Health check endpoint for monitoring.

#### `POST /predict`
Predict churn probability for a user.

**Request Body:**
```json
{
  "age": 35,
  "tenure_days": 180,
  "subscription_tier": "premium",
  "monthly_spend": 75.50,
  "total_interactions": 15,
  "avg_duration": 450.5,
  "resolution_rate": 0.85,
  "recency_days": 7,
  "interaction_frequency": 0.5,
  "sentiment_positive": 8,
  "sentiment_neutral": 5,
  "sentiment_negative": 2
}
```

**Response:**
```json
{
  "churn_probability": 0.73,
  "churn_prediction": true,
  "risk_level": "high",
  "top_risk_factors": [...],
  "recommendation": "..."
}
```

#### `GET /feature-importance`
Get feature importance from the trained model.

**Query Parameters:**
- `top_n` (optional): Number of top features to return (default: 20)

## 🎓 Model Training

### Training Pipeline

1. **Data Generation**
   ```bash
   python scripts/generate_data.py
   ```
   Generates synthetic customer and interaction data.

2. **Feature Engineering**
   ```bash
   python -c "from src.data.preprocessor import ChurnPreprocessor; ..."
   ```
   Creates behavioral, sentiment, and NLP features.

3. **Model Training**
   ```bash
   python scripts/train_model.py
   ```
   Trains Random Forest classifier with SHAP explainer.

### Model Configuration

Edit `src/config.py` to customize:

```python
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "n_estimators": 100,
    "max_depth": 10,
}
```

### Feature Engineering

The system engineers multiple feature types:

- **Behavioral Features**: Interaction frequency, recency, tenure
- **Sentiment Features**: Positive/negative/neutral interaction counts
- **Engagement Features**: Resolution rate, average duration
- **NLP Features**: Text embeddings from customer interactions (optional)

## 🐳 Deployment

### Docker Deployment

1. **Using the deployment script**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

2. **Manual Docker deployment**
   ```bash
   # Build and start containers
   docker-compose up --build -d
   
   # View logs
   docker-compose logs -f
   
   # Stop containers
   docker-compose down
   ```

### Production Deployment

For production environments:

1. **Set environment variables**
   ```bash
   export LOG_LEVEL=WARNING
   export API_HOST=0.0.0.0
   export API_PORT=8000
   ```

2. **Use production-grade WSGI server**
   ```bash
   gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

3. **Set up reverse proxy** (nginx example)
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## 📁 Project Structure

```
context-aware-churn-model/
├── .github/
│   └── workflows/          # CI/CD workflows
│       ├── ci.yml          # Continuous Integration
│       ├── deploy.yml      # Continuous Deployment
│       └── lint.yml        # Code linting
├── data/
│   ├── raw/                # Raw data files
│   └── processed/          # Processed features
├── docs/                   # Additional documentation
├── models/
│   └── saved/              # Trained model artifacts
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/
│   ├── generate_data.py    # Data generation script
│   └── train_model.py      # Model training script
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py         # FastAPI application
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_generator.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── churn_model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── __init__.py
│   └── config.py
├── tests/                  # Unit and integration tests
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.sh               # Setup script
├── deploy.sh              # Deployment script
└── README.md
```

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Write docstrings for all functions and classes
- Maintain test coverage above 80%

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Godfrey Njoro** - [GodfreyNjoro](https://github.com/GodfreyNjoro)

## 🙏 Acknowledgments

- SHAP library for explainability features
- FastAPI for the excellent web framework
- Sentence-Transformers for NLP embeddings
- scikit-learn for machine learning utilities

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact: godfreynjorogewamwere@gmail.com

## 🗺️ Roadmap

- [ ] Add deep learning models (LSTM, Transformer)
- [ ] Implement real-time streaming predictions
- [ ] Add A/B testing framework
- [ ] Integrate with popular CRM systems
- [ ] Add multi-language support
- [ ] Implement automated retraining pipeline
- [ ] Add model versioning and experiment tracking

---

**Note**: This is a demonstration project with synthetic data. For production use, replace the data generation with your actual customer data and retrain the model accordingly.
