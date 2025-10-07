from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GOOGLE_DIRECTIONS_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_POI_API_URL = "https://places.googleapis.com/v1/places:searchText"
