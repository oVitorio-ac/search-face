# Search Face 

Facial recognition search application to find people in event photos using the `face_recognition` library.

## Overview

This project extracts 128-dimensional face embeddings from images and performs similarity searches to identify matching faces across a photo collection.

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── get_vector.py    # Extract face embeddings from a single image
│   └── find_matches.py  # Search for matching faces in a photo folder
├── Pipfile
└── requirements.txt
```

## Installation

```bash
cd backend
pipenv install
pipenv shell
```

## Usage

### Extract Face Vector

Extract facial embeddings from an image:

```bash
python -m app.get_vector --file /path/to/photo.jpg [--model hog|cnn]
```

Output (JSON):
```json
{
  "num_faces": 1,
  "encodings": [[0.123, -0.456, ...]]
}
```

### Find Matches

Search for faces matching a query image:

```bash
python -m app.find_matches \
    --query /path/to/query.jpg \
    --folder /path/to/gallery \
    [--threshold 0.6] \
    [--top 5] \
    [--model hog]
```

Output (JSON):
```json
{
  "query": "/path/to/query.jpg",
  "matches": [
    {"filename": "photo1.jpg", "distance": 0.3123},
    {"filename": "photo2.png", "distance": 0.4157}
  ]
}
