# BloomIQ: Flower Recognition & Agentic RAG Chatbot

BloomIQ is an integrated botanical platform containing two independent subsystems:
1. **Fine-Grained Flower Classifier:** An efficient PyTorch vision pipeline built on MobileNetV2 with Test-Time Augmentation (TTA) and Out-of-Distribution (OOD) guarding.
2. **Agentic RAG Chatbot:** An intelligent botanical assistant that classifies query intents, fetches live web/Wikipedia context, and synthesizes answers using LLMs with a local deterministic backup parser.

---

##  Setup & Installation

### 1. Environment Activation
Activate the conda environment configured for the project:
```bash
conda activate flower_ai
```
*(If the environment does not exist, initialize a Python 3.10 environment and run `pip install -r requirements.txt`)*

### 2. Environment Variables Configuration
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Open `.env` and fill in your API keys:
* `GEMINI_API_KEY`: Primary generative API key.
* `GROQ_API_KEY` or `OPENAI_API_KEY`: Backup LLM endpoints.

---

##  Running the Project

### 1. Launch the FastAPI Backend Server
Run the application server using the python interpreter:
```bash
python app/main.py
```
Upon startup, the server automatically loads the pre-trained model weights (`models/flower_model_v2.pth`) and initializes the class label mappings.

### 2. Access the Web Interface
Once the server is running, open your web browser and navigate to:
```url
http://localhost:8000
```
Through the UI, you can:
* Upload flower images to run species recognition (Task 1).
* Chat with the botanical assistant (Task 2).

---

##  Utilities & Tests

### Run Unit Tests
To verify the chatbot and prediction logic, run the test suite:
```bash
PYTHONPATH=. pytest tests/
```

### View Reports
Visual figures (confusion matrix, class accuracy charts, and training history curves) are located in the `reports/` folder.
