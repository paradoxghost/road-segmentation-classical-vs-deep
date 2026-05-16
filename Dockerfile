FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Install demo-only dependencies. The full requirements file includes notebooks
# and training tools that are unnecessary for running the Streamlit app.
COPY requirements-demo.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir -r requirements-demo.txt

COPY app ./app
COPY src ./src
COPY results/models/unet_best.pt ./results/models/unet_best.pt
COPY results/figures ./results/figures

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/demo_app.py"]
