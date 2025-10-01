from pydantic import BaseModel
from typing import Optional, Dict, Any
from pydantic import BaseModel


# Define request body model
class ChatRequest(BaseModel):
    message: str


class QueryResponse(BaseModel):
    summary: str
    geojson: Dict[str, Any]
    sql_query: Optional[str] = None
