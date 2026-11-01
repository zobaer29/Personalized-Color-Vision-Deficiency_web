"""
FastAPI Main Application Entry Point for Personalized CVD Browser Assistant.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
from .config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for Multi-Type Color Vision Deficiency Recoloring and Personalization.",
)

# Enable CORS for Chrome Extension requests and local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"chrome-extension://[a-p]{32}",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "supported_conditions": ["Protanopia (0)", "Deuteranopia (1)", "Tritanopia (2)"],
    }
