import os
import shutil
import asyncio # 1. Added for concurrency
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import uvicorn
from fastapi.concurrency import run_in_threadpool # 2. Added to handle sync functions in async loop

from agents.agent_01_parser import extract_text_from_pdf, parse_cv_with_gemini

app = FastAPI(title="RecruitMind AI API", version="1.0.0")
TEMP_DIR = "temp_cvs"
os.makedirs(TEMP_DIR, exist_ok=True)

# 3. HELPER FUNCTION: Processes a single CV
async def process_single_cv(file: UploadFile):
    file_path = os.path.join(TEMP_DIR, file.filename)
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run sync functions in threadpool so they don't block the event loop
        raw_text = await run_in_threadpool(extract_text_from_pdf, file_path)
        
        if not raw_text:
            return {"filename": file.filename, "status": "failed", "error": "Text extraction failed"}

        parsed_data = await run_in_threadpool(parse_cv_with_gemini, raw_text)
        
        if parsed_data:
            return {"filename": file.filename, "status": "success", "data": parsed_data.model_dump()}
        return {"filename": file.filename, "status": "failed", "error": "LLM parsing failed"}
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/v1/upload-cvs/")
async def upload_cvs(files: List[UploadFile] = File(...)):
    accepted_formats = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    
    # Validation check
    for file in files:
        if file.content_type not in accepted_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {file.filename}")

    # 4. CONCURRENCY: Start all parsing tasks simultaneously
    tasks = [process_single_cv(file) for file in files]
    processed_results = await asyncio.gather(*tasks)

    return {
        "message": f"Processed {len(files)} CV(s) in parallel.",
        "results": processed_results
    }
