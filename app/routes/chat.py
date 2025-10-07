from fastapi import HTTPException
from app.utils.helper import (get_router_agent_graph)
from app.models.schema import ChatRequest, QueryResponse
from fastapi import APIRouter
from langchain_core.messages import HumanMessage, ToolMessage
import json

router = APIRouter()


@router.post("/")
async def chat_request(request: ChatRequest):
    """
    Router agent decides between BigQuery GIS query, Google Maps route/POI, or simple LLM response.
    Returns unified QueryResponse (adapts for non-spatial cases).
    """
    try:
        # Get the pre-compiled LangGraph agent
        graph = get_router_agent_graph()

        # Invoke the agent with the user message
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
        }
        config = {"recursion_limit": 5}  # Prevent loops
        result = graph.invoke(initial_state, config=config)
        print("result: ", result)
        # Extract final output from state
        output = result.get("output", {})
        if not output and result["messages"]:
            # Fallback: Last message or tool result
            last_msg = result["messages"][-1]
            if hasattr(last_msg, 'content'):
                content = last_msg.content
                # Key fix: Join list to string if content is list (Gemini quirk)
                if isinstance(content, list):
                    content = ''.join(content)
                output = {"response": content, "summary": content}
            elif hasattr(last_msg, 'additional_kwargs') and 'function_call' in last_msg.additional_kwargs:
                # Handle direct tool calls
                output = {
                    "response": f"Tool invoked: {last_msg.additional_kwargs['function_call']}",
                    "summary": f"Tool invoked: {last_msg.additional_kwargs['function_call']}"
                }
            # Additional fallback: Check for ToolMessage in messages
            for msg in reversed(result["messages"]):
                if isinstance(msg, ToolMessage) and msg.content:
                    try:
                        tool_output = json.loads(msg.content)
                        output = tool_output
                        break
                    except json.JSONDecodeError:
                        output = {"response": msg.content,
                                  "summary": msg.content}
                        break

        if not output:
            raise HTTPException(
                status_code=404, detail="No response generated")

        if "geojson" in output:
            summary = output.get(
                "summary", f"Found {len(output['geojson']['features'])} result(s)")
            return QueryResponse(
                summary=summary,
                geojson=output["geojson"],
                sql_query=output.get("sql_query", "")
            )
        elif "route_data" in output:
            # Convert route + POIs to FeatureCollection
            route_coords = output["route_data"]["route_coords"]
            pois = output["route_data"].get("pois", [])
            distance_km = output["route_data"].get("distance_km", 0)
            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in route_coords]},
                        "properties": {"type": "route", "distance_km": distance_km}
                    }
                ] + [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [poi["lon"], poi["lat"]]},
                        "properties": {"name": poi["name"], "distance_km": poi["distance_km"], "address": poi.get("address", "")}
                    }
                    for poi in pois
                ]
            }
            summary = f"Route with {len(pois)} POIs: {request.message}"
            return QueryResponse(
                summary=summary,
                geojson=geojson,
                sql_query=""
            )
        else:
            # Simple chat (key: Use joined string for summary)
            response_text = output.get(
                "summary", "General response generated")
            if isinstance(response_text, list):
                response_text = ''.join(response_text)  # Double-check join
            return QueryResponse(
                summary=response_text,  # Now always str
                geojson={"type": "FeatureCollection", "features": []},
                sql_query=""
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )
