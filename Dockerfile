FROM python:3.11-slim

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y python3-dev && \
    apt-get clean && \ 
    apt-get install -y curl && \
    apt-get install -y unzip && \
    rm -rf /var/lib/apt/lists/* 
    
COPY requirements.txt .

RUN pip install --no-cache-dir h5py --only-binary h5py

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/temp && \
    curl -o /app/temp/tourism_rating.csv https://storage.googleapis.com/toursantara/model-recommendation/tourism_rating.csv && \
    curl -o /app/temp/tourism_with_id.csv https://storage.googleapis.com/toursantara/model-recommendation/tourism_with_id.csv && \
    curl -o /app/temp/model.zip https://storage.googleapis.com/toursantara/model-recommendation/model_recommender.zip && \
    unzip /app/temp/model.zip -d /app/temp/model && \
    rm /app/temp/model.zip

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
