# Search Face

Face recognition API using Flask + `face_recognition` to detect faces, match known people, and return polygon coordinates with best-match names.

## Overview

The backend now exposes HTTP endpoints for recognition:
- `POST /recognize`: compare a query image against known images sent in the same request.
- `POST /recognize-from-folder`: compare a query image against images in a server folder.

Response includes, for each detected face in query image:
- polygon: `top`, `right`, `bottom`, `left`
- best matched `name` (or `"unknown"`)
- `distance`

## Project Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── wsgi.py
│   ├── application/
│   ├── domain/
│   ├── infrastructure/
│   ├── interface/
│   ├── get_vector.py    # Optional CLI utility
│   └── find_matches.py  # Optional CLI utility
├── tests/
├── Pipfile
└── Pipfile.lock
```

## Installation

```bash
cd backend
pipenv install
```

## Run API

### Development (Flask)

```bash
cd backend
pipenv run flask --app app:create_app run --debug
```

### Production style (Gunicorn)

```bash
cd backend
pipenv run gunicorn app.wsgi:app -b 0.0.0.0:5000
```

### Health check

```bash
curl http://127.0.0.1:5000/health
```

## API Usage

### 1) `POST /recognize`

Compare one query image against known images sent by client.

Request (`multipart/form-data`):
- `query_image`: file
- `known_images`: repeated file field
- `known_names`: repeated text field (same order and count as `known_images`)

Example:

```bash
curl -X POST http://127.0.0.1:5000/recognize \
  -F "query_image=@/real/path/query.jpg" \
  -F "known_images=@/real/path/alice.jpg" \
  -F "known_images=@/real/path/bob.jpg" \
  -F "known_names=alice" \
  -F "known_names=bob"
```

### 2) `POST /recognize-from-folder`

Compare one query image against all supported images inside a server folder.

Request (`multipart/form-data`):
- `query_image`: file
- `folder_path`: absolute or relative path on server

Example:

```bash
curl -X POST http://127.0.0.1:5000/recognize-from-folder \
  -F "query_image=@/real/path/query.jpg" \
  -F "folder_path=/real/path/known_people"
```

### Response shape

```json
{
  "faces": [
    {
      "polygon": {"top": 10, "right": 120, "bottom": 80, "left": 40},
      "name": "alice",
      "distance": 0.34211
    }
  ]
}
```

## Error behavior

- `400`: invalid input (missing fields, mismatch lengths, invalid folder/image)
- `422`: no face detected in query image
- `500`: internal server error

## Tests

```bash
cd backend
.venv/bin/python -m unittest discover -s tests
```

## Optional CLI Utilities

These are still available for local debugging/manual workflows:

### Extract vectors

```bash
cd backend
python -m app.get_vector --file /path/to/photo.jpg --model hog
```

### Find matches in folder

```bash
cd backend
python -m app.find_matches --query /path/to/query.jpg --folder /path/to/gallery --threshold 0.6
```
