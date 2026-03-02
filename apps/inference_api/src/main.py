import asyncio  # added for asynchronous execution
import logging
import random
import string
import time
from collections import deque
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

from src.Config import Config
from src.auth import verify_api_key
from src.engines import BiologyInferenceEngine, MaterialsInferenceEngine
from src.models import BiologyRequest, MaterialsRequest, BatchBiologyRequest, BatchMaterialsRequest

app = FastAPI(title="Nexa API", version="2.0.0")
logger = logging.getLogger("Nexa")
logging.basicConfig(level=logging.INFO)

# Load configuration
try:
    config = Config(config_file='config.json')
    MODEL_PATHS = config.get("model_paths")
except FileNotFoundError:
    logger.error("config.json not found. Please create it. Exiting.")
    exit()


engines = {}
for model_type in ["bio_1", "bio_2", "mat_1", "mat_2"]:
    try:
        domain, version = model_type.split("_")
        path_key = "bio" if domain == "bio" else "materials"
        model_path = MODEL_PATHS.get(path_key, {}).get(version)
        if not model_path:
            raise FileNotFoundError(f"Path for {model_type} not found in config.json")

        if domain == "bio":
            engines[model_type] = BiologyInferenceEngine(model_path)
        else:
            engines[model_type] = MaterialsInferenceEngine(model_path)
    except Exception as e:
        logger.error(f"Failed to load {model_type} model: {str(e)}")
        engines[model_type] = None

latency_metrics = {
    "bio": deque(maxlen=100),
    "materials": deque(maxlen=100)
}
request_counts = {
    "bio": 0,
    "materials": 0
}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-ms"] = str(round(process_time, 2))
    path = request.url.path
    if path.startswith("/api/predict/bio"):
        latency_metrics["bio"].append(process_time)
        request_counts["bio"] += 1
    elif path.startswith("/api/predict/materials"):
        latency_metrics["materials"].append(process_time)
        request_counts["materials"] += 1
    return response

def get_avg_latency(endpoint: str):
    values = latency_metrics[endpoint]
    return round(sum(values) / len(values), 2) if values else 0.0

def random_sequence(length=16):
    return ''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=length))

def random_structure(length=8):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "bio": list(MODEL_PATHS["bio"].keys()),
            "materials": list(MODEL_PATHS["materials"].keys())
        }
    }

@app.get("/metrics")
async def get_metrics(_=Depends(verify_api_key)):
    return {
        "bio_avg_latency_ms": get_avg_latency("bio"),
        "materials_avg_latency_ms": get_avg_latency("materials"),
        "bio_requests": request_counts["bio"],
        "materials_requests": request_counts["materials"]
    }

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return """
    <html>
        <head>
            <title>Nexa Model Backend Dashboard</title>
            <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #222; }
                .container { max-width: 950px; margin: 0 auto; padding: 30px; }
                .section { background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.09); margin: 30px 0; padding: 30px; }
                h1, h2 { color: #007bff; }
                .metrics-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                .metrics-table th, .metrics-table td { border: 1px solid #e0e0e0; padding: 8px 12px; text-align: left; }
                .metrics-table th { background: #f7f7f7; }
                .json-output { background: #222; color: #fff; border-radius: 6px; padding: 16px; font-family: 'Fira Mono', monospace; font-size: 1em; overflow-x: auto; margin-top: 10px; }
                button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin-left: 8px; display: inline-flex; align-items: center; }
                button:hover { background: #0056b3; }
                .test-section { display: flex; gap: 30px; flex-wrap: wrap; }
                .test-card { flex: 1 1 300px; min-width: 300px; background: #f8fafc; border-radius: 8px; padding: 18px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
                .dataset-section { margin-top: 30px; background: #f7f7fa; border-radius: 12px; padding: 30px; }
                .dataset-form-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
                .dataset-form-row label { min-width: 140px; }
                .dataset-form-row input[type=number] { width: 80px; padding: 6px; border-radius: 4px; border: 1px solid #ccc; }
                .download-icon { vertical-align: middle; margin-left: 6px; }
                textarea { width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #ccc; min-height: 60px; margin-top: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Nexa Model Backend Dashboard </h1>
                <div class="section">
                    <h2>System Metrics</h2>
                    <table class="metrics-table">
                        <tr>
                            <th>Model</th>
                            <th>Avg Latency (ms)</th>
                            <th>Requests</th>
                        </tr>
                        <tr>
                            <td>NexaBio_1</td>
                            <td id="bio1-latency">0</td>
                            <td id="bio1-req">0</td>
                        </tr>
                        <tr>
                            <td>NexaBio_2</td>
                            <td id="bio2-latency">0</td>
                            <td id="bio2-req">0</td>
                        </tr>
                        <tr>
                            <td>NexaMat_1</td>
                            <td id="mat1-latency">0</td>
                            <td id="mat1-req">0</td>
                        </tr>
                        <tr>
                            <td>NexaMat_2</td>
                            <td id="mat2-latency">0</td>
                            <td id="mat2-req">0</td>
                        </tr>
                    </table>
                </div>
                <div class="section test-section">
                    <div class="test-card">
                        <h2>NexaBio_1 (Secondary Structure)</h2>
                        <button onclick="testBio1()">Test Secondary Structure</button>
                        <div id="bio1Result" class="json-output"></div>
                    </div>
                    <div class="test-card">
                        <h2>NexaBio_2 (Tertiary Structure)</h2>
                        <button onclick="testBio2()">Test Tertiary Structure</button>
                        <div id="bio2Result" class="json-output"></div>
                    </div>
                    <div class="test-card">
                        <h2>NexaMat_1 (Materials Prediction)</h2>
                        <button onclick="testMat1()">Test Materials Prediction</button>
                        <div id="mat1Result" class="json-output"></div>
                    </div>
                </div>
                <div class="section">
                    <h2>End-to-End SciML Pipeline</h2>
                    <div class="test-card">
                        <h3>Biology Pipeline (Drug Discovery)</h3>
                        <textarea id="bio-pipeline-query" placeholder="Enter a query, e.g., 'Generate hypotheses for protein folding under high temperature'"></textarea>
                        <button onclick="runBioPipeline()">Run Pipeline</button>
                        <div id="bioPipelineResult" class="json-output"></div>
                    </div>
                </div>
                <div class="dataset-section">
                    <h2>Generate & Download Synthetic Dataset</h2>
                    <div class="dataset-form-row">
                        <label for="bio-dataset-size">Bio Dataset Size:</label>
                        <input type="number" id="bio-dataset-size" value="10" min="1" max="1000"/>
                        <button onclick="generateDataset('bio')">Preview JSON</button>
                        <button onclick="downloadCSV('bio')">
                            <span>Download CSV</span>
                            <svg class="download-icon" width="18" height="18" viewBox="0 0 20 20" fill="none">
                                <path d="M10 3v10m0 0l-4-4m4 4l4-4M4 17h12" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </button>
                    </div>
                    <div id="bioDatasetResult" class="json-output"></div>
                    <div class="dataset-form-row">
                        <label for="mat-dataset-size">Materials Dataset Size:</label>
                        <input type="number" id="mat-dataset-size" value="10" min="1" max="1000"/>
                        <button onclick="generateDataset('materials')">Preview JSON</button>
                        <button onclick="downloadCSV('materials')">
                            <span>Download CSV</span>
                            <svg class="download-icon" width="18" height="18" viewBox="0 0 20 20" fill="none">
                                <path d="M10 3v10m0 0l-4-4m4 4l4-4M4 17h12" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </button>
                    </div>
                    <div id="matDatasetResult" class="json-output"></div>
                </div>
            </div>
            <script>
                function syntaxHighlight(json) {
                    if (typeof json != 'string') json = JSON.stringify(json, null, 2);
                    json = json.replace(/\[\s*([\d\.,\s-]+)\s*\]/g, m => m.replace(/\s+/g, ''));
                    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
                        let cls = 'number';
                        if (/^"/.test(match)) cls = /:$/.test(match) ? 'key' : 'string';
                        else if (/true|false/.test(match)) cls = 'boolean';
                        else if (/null/.test(match)) cls = 'null';
                        return '<span style="color:' +
                            (cls === 'key' ? '#7ec699' :
                            cls === 'string' ? '#d19a66' :
                            cls === 'number' ? '#61afef' :
                            cls === 'boolean' ? '#e06c75' :
                            cls === 'null' ? '#c678dd' : '#fff') +
                            '">' + match + '</span>';
                    });
                }

                async function updateMetrics() {
                    const res = await fetch('/metrics', { headers: { 'X-API-Key': 'development_key' } });
                    const data = await res.json();
                    document.getElementById('bio1-latency').innerText = data.bio_avg_latency_ms;
                    document.getElementById('bio2-latency').innerText = data.bio_avg_latency_ms;
                    document.getElementById('mat1-latency').innerText = data.materials_avg_latency_ms;
                    document.getElementById('mat2-latency').innerText = data.materials_avg_latency_ms;
                    document.getElementById('bio1-req').innerText = data.bio_requests;
                    document.getElementById('bio2-req').innerText = data.bio_requests;
                    document.getElementById('mat1-req').innerText = data.materials_requests;
                    document.getElementById('mat2-req').innerText = data.materials_requests;
                }
                setInterval(updateMetrics, 2000);
                window.onload = updateMetrics;

                function randomSeq(len) {
                    const chars = 'ACDEFGHIKLMNPQRSTVWY';
                    let seq = '';
                    for (let i = 0; i < len; i++) seq += chars.charAt(Math.floor(Math.random() * chars.length));
                    return seq;
                }
                function randomStruct(len) {
                    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
                    let seq = '';
                    for (let i = 0; i < len; i++) seq += chars.charAt(Math.floor(Math.random() * chars.length));
                    return seq;
                }

                async function testBio1() {
                    const seq = randomSeq(16);
                    const res = await fetch('/api/predict/bio', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-API-Key': 'development_key' },
                        body: JSON.stringify({ sequence: seq, model_version: '1', confidence_threshold: 0.8, explain: true })
                    });
                    const data = await res.json();
                    document.getElementById('bio1Result').innerHTML = '<pre>' + syntaxHighlight(data) + '</pre>';
                    updateMetrics();
                }
                async function testBio2() {
                    const seq = randomSeq(16);
                    const res = await fetch('/api/predict/bio', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-API-Key': 'development_key' },
                        body: JSON.stringify({ sequence: seq, model_version: '2', confidence_threshold: 0.8, mode: 'speculative' })
                    });
                    const data = await res.json();
                    document.getElementById('bio2Result').innerHTML = '<pre>' + syntaxHighlight(data) + '</pre>';
                    updateMetrics();
                }
                async function testMat1() {
                    const struct = randomStruct(8);
                    const res = await fetch('/api/predict/materials', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-API-Key': 'development_key' },
                        body: JSON.stringify({ structure: struct, model_version: '1', energy_threshold: 0.5, explain: true })
                    });
                    const data = await res.json();
                    document.getElementById('mat1Result').innerHTML = '<pre>' + syntaxHighlight(data) + '</pre>';
                    updateMetrics();
                }

                async function runBioPipeline() {
                    const query = document.getElementById('bio-pipeline-query').value;
                    if (!query) {
                        alert('Please enter a query for the pipeline.');
                        return;
                    }
                    const res = await fetch('/api/pipeline/bio', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-API-Key': 'development_key' },
                        body: JSON.stringify({ query: query })
                    });
                    const data = await res.json();
                    document.getElementById('bioPipelineResult').innerHTML = '<pre>' + syntaxHighlight(data) + '</pre>';
                }

                async function generateDataset(type) {
                    let size = 10, url = '', resultDiv = '';
                    if (type === 'bio') {
                        size = document.getElementById('bio-dataset-size').value;
                        url = '/api/dataset/bio';
                        resultDiv = 'bioDatasetResult';
                    } else {
                        size = document.getElementById('mat-dataset-size').value;
                        url = '/api/dataset/materials';
                        resultDiv = 'matDatasetResult';
                    }
                    const res = await fetch(url + '?size=' + size, {
                        method: 'POST',
                        headers: { 'X-API-Key': 'development_key' }
                    });
                    const data = await res.json();
                    document.getElementById(resultDiv).innerHTML = '<pre>' + syntaxHighlight(data) + '</pre>';
                }

                async function downloadCSV(type) {
                    let size = 10, url = '', filename = '';
                    if (type === 'bio') {
                        size = document.getElementById('bio-dataset-size').value;
                        url = '/api/dataset/bio/csv?size=' + size;
                        filename = 'bio_dataset.csv';
                    } else {
                        size = document.getElementById('mat-dataset-size').value;
                        url = '/api/dataset/materials/csv?size=' + size;
                        filename = 'materials_dataset.csv';
                    }
                    const res = await fetch(url, { method: 'POST', headers: { 'X-API-Key': 'development_key' } });
                    const blob = await res.blob();
                    const link = document.createElement('a');
                    link.href = window.URL.createObjectURL(blob);
                    link.download = filename;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }
            </script>
        </body>
    </html>
    """

@app.post("/api/predict/bio")
async def predict_bio(request: BiologyRequest, _=Depends(verify_api_key)):
    try:
        engine = engines.get(f"bio_{request.model_version}")
        if engine is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        raw_result = await asyncio.to_thread(
            engine.predict,
            {
                "sequence": request.sequence,
                "confidence_threshold": request.confidence_threshold,
                "mode": request.mode,
                "explain": request.explain
            }
        )
        if "tertiary_coordinates" in raw_result:
            raw_result["tertiary_coordinates"] = [
                [float(f"{x[0]:.2f}"), float(f"{x[1]:.2f}"), float(f"{x[2]:.2f}")]
                for x in raw_result["tertiary_coordinates"]
            ]
        result = {"model": f"NexaBio_{request.model_version}", **raw_result}
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Biology prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/materials")
async def predict_materials(request: MaterialsRequest, _=Depends(verify_api_key)):
    try:
        engine = engines.get(f"mat_{request.model_version}")
        if engine is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        raw_result = await asyncio.to_thread(
            engine.predict,
            {
                "structure": request.structure,
                "energy_threshold": request.energy_threshold,
                "mode": request.mode,
                "explain": request.explain
            }
        )
        result = {"model": f"NexaMat_{request.model_version}", **raw_result}
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Materials prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/bio/batch")
async def predict_bio_batch(request: BatchBiologyRequest, _=Depends(verify_api_key)):
    async def process_req(item):
        try:
            engine = engines.get(f"bio_{item.model_version}")
            if engine is None:
                return {"error": "Model not loaded"}
            raw_result = await asyncio.to_thread(
                engine.predict,
                {
                    "sequence": item.sequence,
                    "confidence_threshold": item.confidence_threshold,
                    "mode": item.mode,
                    "explain": item.explain
                }
            )
            return {"model": f"NexaBio_{item.model_version}", **raw_result}
        except Exception as e:
            return {"error": str(e)}
    tasks = [process_req(item) for item in request.requests]
    results = await asyncio.gather(*tasks)
    return JSONResponse(content=results)

@app.post("/api/predict/materials/batch")
async def predict_materials_batch(request: BatchMaterialsRequest, _=Depends(verify_api_key)):
    async def process_req(item):
        try:
            engine = engines.get(f"mat_{item.model_version}")
            if engine is None:
                return {"error": "Model not loaded"}
            raw_result = await asyncio.to_thread(
                engine.predict,
                {
                    "structure": item.structure,
                    "energy_threshold": item.energy_threshold,
                    "mode": item.mode,
                    "explain": item.explain
                }
            )
            return {"model": f"NexaMat_{item.model_version}", **raw_result}
        except Exception as exc:
            tasks = [process_req(item) for item in request.requests]
            results = await asyncio.gather(*tasks)
            return JSONResponse(content=results)
    @app.post("/api/pipeline/bio")
    async def bio_pipeline(request: dict, _=Depends(verify_api_key)):
        # Simplified pipeline implementation
        try:
            query = request.get("query", "")
            # Step 1: Generate candidate sequences
            sequences = [random_sequence() for _ in range(3)]

            # Step 2: Predict structures for each sequence
            predictions = []
            for seq in sequences:
                engine = engines.get("bio_1")
                if engine:
                    raw_result = await asyncio.to_thread(
                        engine.predict,
                        {"sequence": seq, "confidence_threshold": 0.7, "explain": True}
                    )
                    predictions.append({"sequence": seq, "prediction": raw_result})

            return {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "candidates": predictions
            }
        except Exception as e:
            logger.error(f"Bio pipeline failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/dataset/bio")
    async def generate_bio_dataset(size: int = 10, _=Depends(verify_api_key)):
        data = []
        for _ in range(size):
            seq = random_sequence()
            data.append({"sequence": seq, "id": f"BIO_{time.time()}_{random.randint(1000, 9999)}"})
        return data

    @app.post("/api/dataset/materials")
    async def generate_materials_dataset(size: int = 10, _=Depends(verify_api_key)):
        data = []
        for _ in range(size):
            struct = random_structure()
            data.append({"structure": struct, "id": f"MAT_{time.time()}_{random.randint(1000, 9999)}"})
        return data

    @app.post("/api/dataset/bio/csv")
    async def generate_bio_csv(size: int = 10, _=Depends(verify_api_key)):
        data = ["id,sequence,predicted_structure"]
        for _ in range(size):
            seq = random_sequence()
            data.append(f"BIO_{time.time()}_{random.randint(1000, 9999)},{seq},{''.join(random.choices('HEC', k=len(seq)))}")
        return "\n".join(data)

    @app.post("/api/dataset/materials/csv")
    async def generate_materials_csv(size: int = 10, _=Depends(verify_api_key)):
        data = ["id,structure,energy,stability"]
        for _ in range(size):
            struct = random_structure()
            energy = round(random.uniform(-10.0, 0.0), 2)
            stability = random.choice(["high", "medium", "low"])
            data.append(f"MAT_{time.time()}_{random.randint(1000, 9999)},{struct},{energy},{stability}")
        return "\n".join(data)

    if __name__ == "__main__":
        import uvicorn
        logger.info("Starting Nexa API server...")
        logger.info(f"Loaded models: {[k for k, v in engines.items() if v is not None]}")
        uvicorn.run(app, host="0.0.0.0", port=8000)
