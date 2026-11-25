# AI IT Support Chatbot
A beginner NLP retrieval project using TF-IDF and cosine similarity. It retrieves predefined troubleshooting responses, not generative AI answers.

## Run on macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open http://localhost:8501. Test WiFi not connecting, Forgot password, Laptop is slow, and unrelated queries. The 0.30 similarity threshold is a heuristic; expand the dataset and validate before real-world use. Never enter sensitive information.
