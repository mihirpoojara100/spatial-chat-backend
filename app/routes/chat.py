from fastapi import HTTPException
from app.utils.helper import bigquery_to_geojson, get_bigquery_client, get_llm_chain
from app.models.schema import ChatRequest, QueryResponse
import os
import re
from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def chat_request(request: ChatRequest):
    """
    Convert natural language to BigQuery GIS query and return GeoJSON
    """
    try:
        project_id = os.getenv("BIGQUERY_PROJECT_ID")
        dataset_id = os.getenv("BIGQUERY_DATASET_ID")

        llm_chain = get_llm_chain()

        sql_response = llm_chain.invoke({
            "query": request.message,
            "project_id": project_id,
            "dataset_id": dataset_id
        })

        print("sql_response", sql_response.content)

        # Extract raw SQL text
        sql_query = sql_response.content.strip()
        sql_query = re.sub(r"^```(?:sql)?", "", sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r"```$", "", sql_query, flags=re.MULTILINE).strip()

        # Run on BigQuery
        bq_client = get_bigquery_client()
        query_job = bq_client.query(sql_query)
        results = query_job.result()

        # Convert to GeoJSON
        geojson = bigquery_to_geojson(results)
        if not geojson["features"]:
            raise HTTPException(
                status_code=404,
                detail="No matching data found for the query"
            )

        summary = f"Found {len(geojson['features'])} result(s) for: {request.message}"

        return QueryResponse(
            summary=summary,
            geojson=geojson,
            sql_query=sql_query
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )
