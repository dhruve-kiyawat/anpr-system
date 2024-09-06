# Importing necessary modules
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importing custom modules
from .models.RestfulModel import *
from .routers import anpr
from .utils.RouterFunctions import *


app = FastAPI(
    title="Paddle OCR API",
    description="API using Paddle OCR and FastAPI for OCR operations."
)


# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Including routers
app.include_router(anpr.router)