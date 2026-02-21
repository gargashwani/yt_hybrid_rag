import os, uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

router = APIRouter(
    prefix="/storage",
    tags=["storage"]
)

ACCOUNT_URL="https://codesipsdocs.blob.core.windows.net"
CONTAINER_NAME = "agent-docs"


# Route to upload file
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code = 400, detail="Only PDFs allowed")
    
    unique_filename = f"{uuid.uuid4()}-{file.filename}"
    service_client = BlobServiceClient(ACCOUNT_URL, credential=DefaultAzureCredential())

    try:
        blob_client = service_client.get_blob_client(container=CONTAINER_NAME, blob = unique_filename)
        contents = await file.read()
        blob_client.upload_blob(contents)
        return {"filename": unique_filename, "status": "stored", "container": CONTAINER_NAME}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")