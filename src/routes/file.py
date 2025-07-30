from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import DATA_INPUT_DIR, DATA_OUTPUT_DIR, DATA_TEMP_DIR

router = APIRouter()

DIRS = {
    "input": DATA_INPUT_DIR,
    "output": DATA_OUTPUT_DIR,
    "temp": DATA_TEMP_DIR
}


@router.get("/files/{dir_name}", response_model=List[str])
def list_files(dir_name: str):
    """List files in a data directory (input/output/temp)."""
    data_dir = DIRS.get(dir_name)
    if data_dir is None:
        raise HTTPException(status_code=404, detail="Directory not found")
    try:
        return [f.name for f in data_dir.iterdir() if f.is_file()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{dir_name}/{filename}")
def get_file(dir_name: str, filename: str):
    """Download a file from specified data directory."""
    data_dir = DIRS.get(dir_name)
    if data_dir is None:
        raise HTTPException(status_code=404, detail="Directory not found")
    file_path = data_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    # You may want to restrict to certain file extensions here!
    return FileResponse(path=file_path, filename=filename)
