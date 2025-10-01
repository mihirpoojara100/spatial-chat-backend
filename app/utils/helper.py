from typing import Dict, Any
from google.cloud import bigquery
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import json


def bigquery_to_geojson(results) -> Dict[str, Any]:
    """Convert BigQuery results to GeoJSON FeatureCollection"""
    features = []

    for row in results:
        try:
            row_dict = dict(row)

            # Extract geometry
            geometry = None
            geometry_keys = ['geometry', 'geojson', 'location', 'geo']

            for key in geometry_keys:
                if key in row_dict and row_dict[key]:
                    geometry_str = row_dict[key]
                    if isinstance(geometry_str, str):
                        geometry = json.loads(geometry_str)
                    else:
                        geometry = geometry_str
                    break

            if not geometry:
                continue

            # Exclude geometry fields from properties
            properties = {k: v for k, v in row_dict.items()
                          if k not in geometry_keys and v is not None}

            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties
            }
            features.append(feature)

        except Exception:
            continue

    return {
        "type": "FeatureCollection",
        "features": features
    }


def get_bigquery_client():
    """Initialize BigQuery client with credentials"""
    credentials_path = "credentials.json"
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )
        return bigquery.Client(credentials=credentials)
    return bigquery.Client()


def get_llm_chain():
    """Initialize LangChain with Google Gemini"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key,
        temperature=0.7,
        max_output_tokens=2048
    )

    prompt_template = """You are an expert SQL generator for Google BigQuery GIS queries.

    Convert the following natural language query into a valid BigQuery SQL statement that outputs GeoJSON data.

    REQUIREMENTS:
    1. Use BigQuery GIS functions (ST_AsGeoJSON, ST_Within, ST_Distance, etc.)
    2. Always include ST_AsGeoJSON to convert geometry
    3. Output columns: id, name, properties, geometry
    4. Use spatial predicates properly
    5. Ensure results can be converted to GeoJSON FeatureCollection
    6. Use full table path: `project.dataset.table`
    7. GEOGRAPHY data type for spatial columns

    Available tables:
    - `project.dataset.schools` (id, name, location GEOGRAPHY, type, capacity)
    - `project.dataset.flood_zones` (id, name, geometry GEOGRAPHY, risk_level, area_sq_km)
    - `project.dataset.parks` (id, name, geometry GEOGRAPHY, area_sq_km, facilities)
    - `project.dataset.hospitals` (id, name, location GEOGRAPHY, beds, emergency)

    Natural Language Query: {query}
    Project ID: {project_id}
    Dataset ID: {dataset_id}

    SQL Query:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "project_id", "dataset_id"]
    )

    return prompt | llm
