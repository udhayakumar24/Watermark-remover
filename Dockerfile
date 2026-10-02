FROM python:3.11-slim

# ffmpeg binary + OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the LaMa weights into the image (88 MB, Apache-2.0)
RUN python scripts/fetch_model.py

EXPOSE 8000
# Honor Render's $PORT (falls back to 8000 locally)
CMD ["sh", "-c", "python app.py --host 0.0.0.0 --port ${PORT:-8000}"]
