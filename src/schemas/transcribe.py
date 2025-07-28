from typing import Optional, List

from pydantic import BaseModel

class TranscribeRequest(BaseModel):
    input: str
    language: Optional[str] = None
    need_alignment: Optional[bool] = None
    return_srt: Optional[bool] = False
    return_vtt: Optional[bool] = False


class BatchRequest(BaseModel):
    inputs: List[str]
    language: Optional[str] = None
    need_alignment: Optional[bool] = None
    return_srt: Optional[bool] = False
    return_vtt: Optional[bool] = False