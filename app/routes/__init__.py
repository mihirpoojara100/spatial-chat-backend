from fastapi import APIRouter
from .chat import router as chat_router

api_router = APIRouter()


api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
