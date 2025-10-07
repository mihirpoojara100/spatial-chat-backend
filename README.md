# Spatial Chat Backend

A FastAPI-based backend service that combines LLM-powered chat with geospatial functionality using Google Maps and BigQuery.

## Features

- Route planning with points of interest (POIs) discovery
- Intelligent chat routing using LangGraph and Gemini Pro
- Integration with Google Maps API for routing and places
- CORS-enabled API endpoints
- Environment-based configuration

## Prerequisites

- Python 3.x
- Google Cloud credentials
- Google Maps API key
- Google Gemini API key

## Installation

1. Clone the repository
2. Create a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:

```sh
pip install -r requirements.txt
```

4. Configure environment variables in `.env`:

```ini
GOOGLE_API_KEY="your-gemini-api-key"
GOOGLE_MAPS_API_KEY="your-maps-api-key"
BIGQUERY_PROJECT_ID="your-project-id"
BIGQUERY_DATASET_ID="your-dataset-id"
```

## Project Structure

```
├── app/
│   ├── models/
│   │   └── schema.py         # Pydantic models
│   ├── routes/
│   │   ├── __init__.py      # Router configuration
│   │   └── chat.py          # Chat endpoint handlers
│   └── utils/
│       ├── constants.py      # Environment variables
│       └── helper.py         # LangGraph agent implementation
├── main.py                   # FastAPI application
└── requirements.txt          # Project dependencies
```

## API Endpoints

### POST `/api/chat/`

Main chat endpoint that processes user queries and returns structured responses.

Request body:

```json
{
  "message": "string"
}
```

Response:

```json
{
  "summary": "string",
  "geojson": {
    "type": "FeatureCollection",
    "features": []
  },
  "sql_query": "string"
}
```

## Features in Detail

### Route Planning

- Extracts start and end locations from natural language queries
- Calculates optimal driving routes using Google Maps Directions API
- Optionally searches for POIs along the route
- Returns GeoJSON with route linestring and POI points

### Chat System

- Uses Google's Gemini Pro model for natural language understanding
- Implements a LangGraph-based routing system for query classification
- Supports both general chat and specialized geospatial queries

## Development

Run the development server:

```sh
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Security Notes

- Store sensitive credentials in `.env` (not in version control)
- Review CORS settings in `main.py` for production
- Implement proper API authentication for production use
