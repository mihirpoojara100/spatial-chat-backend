# Need to dig deep in geo points and it's tool
from typing import Dict, Any, TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import json
import requests
import polyline
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
from app.utils.constants import GOOGLE_API_KEY, GOOGLE_MAPS_API_KEY, GOOGLE_DIRECTIONS_API_URL, GOOGLE_POI_API_URL


# Env globals
geolocator = Nominatim(user_agent="gis_agent")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    output: Dict[str, Any]
    project_id: str
    dataset_id: str


@tool
def geocode_place(place_name: str) -> Dict[str, Any]:
    """Geocode a place name to get its location as GeoJSON."""
    def geocode_city(city: str) -> tuple:
        try:
            location = geolocator.geocode(city)
            time.sleep(1)
            if location:
                return (location.latitude, location.longitude), location.address
        except GeocoderTimedOut:
            pass
        return None, None

    point, address = geocode_city(place_name)
    if not point:
        raise ValueError(f"Could not geocode '{place_name}'")

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [point[1], point[0]]  # [lon, lat]
                },
                "properties": {
                    "type": "location",
                    "name": place_name,
                    "address": address or "N/A"
                }
            }
        ]
    }

    return {"geojson": geojson}


@tool
def get_route_with_pois(route_query: str, poi_type: str = "") -> Dict[str, Any]:
    """Get route and POIs. Extracts start/end from query (e.g., 'from Point A to B').
    If poi_type is empty, only the route is returned without searching for POIs."""

    def geocode_city(city: str) -> tuple:
        try:
            location = geolocator.geocode(city)
            time.sleep(1)
            if location:
                return (location.latitude, location.longitude)
        except GeocoderTimedOut:
            pass
        return None

    parts = route_query.lower().split(' to ')
    if len(parts) >= 2:
        start_city = parts[0].replace('from ', '').strip()
        end_city = parts[1].strip()
        point_a = geocode_city(start_city)
        point_b = geocode_city(end_city)
    else:
        raise ValueError("Could not parse start/end from query")

    if not point_a or not point_b:
        raise ValueError("Geocoding failed for cities")

    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
        'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline'
    }
    data = {
        'origin': {'location': {'latLng': {'latitude': point_a[0], 'longitude': point_a[1]}}},
        'destination': {'location': {'latLng': {'latitude': point_b[0], 'longitude': point_b[1]}}},
        'travelMode': 'DRIVE',
        'routingPreference': 'TRAFFIC_AWARE_OPTIMAL',
        'polylineQuality': 'HIGH_QUALITY'
    }

    response = requests.post(GOOGLE_DIRECTIONS_API_URL,
                             headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f"Routes error: {response.text}")

    result = response.json()
    encoded_polyline = result['routes'][0]['polyline']['encodedPolyline']
    route_coords = polyline.decode(encoded_polyline)
    distance_km = result['routes'][0]['distanceMeters'] / 1000

    pois = []
    if poi_type and poi_type.strip():
        poi_headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
            'X-Goog-FieldMask': 'places.displayName,places.location,places.formattedAddress'
        }
        poi_data = {
            'textQuery': poi_type,
            'pageSize': 20,
            'searchAlongRouteParameters': {'polyline': {'encodedPolyline': encoded_polyline}}
        }
        poi_response = requests.post(
            GOOGLE_POI_API_URL, headers=poi_headers, json=poi_data)
        if poi_response.status_code != 200:
            pois = []
        else:
            pois_raw = poi_response.json().get('places', [])
            pois = []
            for p in pois_raw:
                lat = p['location']['latitude']
                lon = p['location']['longitude']
                name = p['displayName']['text']
                address = p.get('formattedAddress', 'N/A')
                min_dist = min(
                    geodesic((lat, lon), coord).km for coord in route_coords)
                pois.append({'name': name, 'lat': lat, 'lon': lon,
                            'distance_km': min_dist, 'address': address})
            pois.sort(key=lambda x: x['distance_km'])

    # Convert to GeoJSON
    route_features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lat, lon in route_coords]
            },
            "properties": {
                "type": "route",
                "distance_km": distance_km
            }
        }
    ]

    poi_features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [poi["lon"], poi["lat"]]
            },
            "properties": {
                "type": "poi",
                "name": poi["name"],
                "distance_km": poi["distance_km"],
                "address": poi["address"]
            }
        }
        for poi in pois[:10]
    ]

    geojson = {
        "type": "FeatureCollection",
        "features": route_features + poi_features
    }

    return {"geojson": geojson}


@tool
def simple_chat_response(query: str) -> str:
    """Direct LLM response for general queries."""
    return llm.invoke([HumanMessage(content=query)]).content


def router_node(state: AgentState) -> Dict[str, Any]:
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify and act:
        - LOCATION: Queries like "where is X situated?", "location of X", "find X on map" → geocode_place(place_name="X"). Extract the place name accurately.
        - ROUTE: Queries like "from A to B", "route from A to B", "POIs like Y between A and B" → get_route_with_pois(route_query="from A to B", poi_type="Y" if specified, else empty).
        - CHAT: All other general queries → simple_chat_response(query=full user query).
        Always call the appropriate tool; extract parameters precisely."""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = router_prompt | llm.bind_tools(
        [geocode_place, get_route_with_pois, simple_chat_response])
    user_msg = HumanMessage(content=state["messages"][-1].content if hasattr(
        state["messages"][-1], 'content') else state["messages"][-1]["content"])
    response = chain.invoke({"messages": [user_msg]})
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    # Guard: Only AIMessages can have tool_calls
    if not isinstance(last_msg, AIMessage):
        return END
    if last_msg.tool_calls:  # Simplified: No hasattr needed if type-checked
        # Optional: Break if last tool was chat to prevent loops
        if len(state["messages"]) >= 2 and hasattr(state["messages"][-2], 'tool_calls') and state["messages"][-2].tool_calls:
            tool_name = state["messages"][-2].tool_calls[0].get('name', '')
            if tool_name == 'simple_chat_response':
                return END
        return "tools"
    return END


def agent_node(state: AgentState) -> Dict[str, Any]:
    # Check if last was chat tool → Use LLM to generate/summarize and end
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'name') and last_msg.name == 'simple_chat_response':
        # LLM generate for chat (moved here to avoid nested calls)
        chat_prompt = ChatPromptTemplate.from_template(
            "Answer this query based on knowledge: {query}. Be concise.")
        chat_chain = chat_prompt | llm
        response = chat_chain.invoke({"query": last_msg.content.split(": ")[
                                     # Extract query from placeholder
                                     1] if ": " in last_msg.content else last_msg.content})
        state["output"] = {"summary": response.content}
        # End with final message
        return {"messages": [AIMessage(content=response.content)]}
    else:
        # Standard ReAct for other tools (location/route)
        tools = [geocode_place, get_route_with_pois, simple_chat_response]
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. 
            After using a location or route tool, summarize the result in natural language: 
            Describe the place/route/POIs clearly, mention key details like address or distance, 
            and suggest viewing on the map. End—no further tools unless needed."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = agent_prompt | llm.bind_tools(tools)
        response = chain.invoke(state["messages"])
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            # Set output from previous tool + LLM summary
            tool_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage) and msg.content:
                    try:
                        tool_output = json.loads(msg.content)
                        tool_msg = tool_output
                        break
                    except json.JSONDecodeError:
                        tool_msg = {"response": msg.content}
                        break
            if tool_msg:
                if "geojson" in tool_msg:
                    state["output"] = {
                        "summary": response.content,
                        "geojson": tool_msg["geojson"]
                    }
                else:
                    state["output"] = {"summary": response.content}
            else:
                state["output"] = {"summary": response.content}
        return {"messages": [response]}


def get_router_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(
        [geocode_place, get_route_with_pois, simple_chat_response]))

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", should_continue, {
                                   "tools": "tools", END: END})  # Already good
    workflow.add_edge("tools", "agent")
    # Key fix: Full mapping for agent conditional
    workflow.add_conditional_edges("agent", should_continue, {
                                   "tools": "tools", END: END})

    return workflow.compile()
