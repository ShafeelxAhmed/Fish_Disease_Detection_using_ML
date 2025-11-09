FROM python:3.10

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y git wget

# Copy app files
COPY backend/ /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install tensorflow==2.10.0 keras==2.10.0 fastapi uvicorn numpy pillow requests h5py

EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
