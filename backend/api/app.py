from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict, optimize, benchmark, scenario

app = FastAPI(
    title="Green Fleet Management Optimization API (SIH 2026)",
    description="Quantum-Inspired Optimization & Machine Learning Prediction Core",
    version="1.0.0"
)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(predict.router)
app.include_router(optimize.router)
app.include_router(benchmark.router)
app.include_router(scenario.router)

@app.get("/")
def root():
    return {
        "service": "Green Fleet Optimizer Python Microservice",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
