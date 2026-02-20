# Azure Blob Storage Flow Diagram

```mermaid
flowchart TD
    Start([API Request]) --> Upload{Upload Document?}
    Start --> List{List Documents?}
    Start --> Extract{Extract Text?}
    
    %% Upload Flow
    Upload -->|POST /upload-doc| ValidatePDF{Validate PDF}
    ValidatePDF -->|Valid| GenerateFilename[Generate Unique Filename]
    GenerateFilename --> GetBlobClient1[Get Blob Client<br/>container: agent-docs]
    GetBlobClient1 --> UploadBlob[Upload Blob<br/>blob_client.upload_blob]
    UploadBlob --> AzureBlob1[(Azure Blob Storage<br/>Container: agent-docs)]
    UploadBlob --> ReturnUpload[Return filename & status]
    
    %% List Flow
    List -->|GET /list-docs| GetServiceClient1[Get Blob Service Client<br/>from connection string]
    GetServiceClient1 --> GetContainerClient[Get Container Client<br/>container: agent-docs]
    GetContainerClient --> ListBlobs[List Blobs<br/>container_client.list_blobs]
    ListBlobs --> AzureBlob2[(Azure Blob Storage<br/>Container: agent-docs)]
    ListBlobs --> FormatResults[Format Results<br/>name, size, created]
    FormatResults --> ReturnList[Return list of documents]
    
    %% Extract Flow
    Extract -->|Called by ingest| GetServiceClient2[Get Blob Service Client<br/>from connection string]
    GetServiceClient2 --> GetBlobClient2[Get Blob Client<br/>container: agent-docs<br/>blob: blob_name]
    GetBlobClient2 --> DownloadBlob[Download Blob<br/>blob_client.download_blob]
    DownloadBlob --> AzureBlob3[(Azure Blob Storage<br/>Container: agent-docs)]
    DownloadBlob --> GetBytes[Get Blob Bytes<br/>readall]
    GetBytes --> CheckBytes{Bytes Exist?}
    CheckBytes -->|No| ReturnNone[Return None]
    CheckBytes -->|Yes| OpenPDF[Open PDF with PyMuPDF<br/>fitz.open]
    OpenPDF --> ExtractText[Extract Text from Pages<br/>page.get_text]
    ExtractText --> ChunkText[Chunk Text<br/>RecursiveCharacterTextSplitter]
    ChunkText --> ReturnChunks[Return Text Chunks]
    
    %% Styling - Black and White Printer Friendly
    classDef apiEndpoint fill:#f5f5f5,stroke:#000000,stroke-width:2px,color:#000000
    classDef storageOp fill:#e0e0e0,stroke:#000000,stroke-width:2px,color:#000000
    classDef azureStorage fill:#d0d0d0,stroke:#000000,stroke-width:3px,color:#000000
    classDef processing fill:#ffffff,stroke:#000000,stroke-width:2px,stroke-dasharray: 5 5,color:#000000
    
    class Upload,List,Extract apiEndpoint
    class GetServiceClient1,GetServiceClient2,GetBlobClient1,GetBlobClient2,GetContainerClient,UploadBlob,DownloadBlob,ListBlobs storageOp
    class AzureBlob1,AzureBlob2,AzureBlob3 azureStorage
    class ValidatePDF,GenerateFilename,GetBytes,CheckBytes,OpenPDF,ExtractText,ChunkText,FormatResults processing
```

## Storage Account Services Used

### 1. **BlobServiceClient**
   - Initialized from `AZURE_STORAGE_CONNECTION_STRING`
   - Entry point for all blob operations

### 2. **Container Client** (`agent-docs`)
   - Operations: `list_blobs()`
   - Returns blob metadata (name, size, created date)

### 3. **Blob Client**
   - Operations:
     - `upload_blob()` - Upload PDF files
     - `download_blob()` - Download blob bytes
   - Container: `agent-docs`
   - Blob naming: `{uuid}-{filename}.pdf`

### 4. **Text Processing** (Storage-related)
   - `extract_text_from_blob()` - PDF text extraction using PyMuPDF
   - `chunk_text()` - Text splitting for processing

## API Endpoints

- **POST /upload-doc** → Uploads PDF to blob storage
- **GET /list-docs** → Lists all blobs in container
- **POST /ingest/{blob_name}** → Uses blob storage to extract and process text
