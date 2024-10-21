from pydantic import BaseModel

class ResultInfoModel(BaseModel):
    FileName:str|None
    FileId:str|None
    TB_InferenceResult:bool = False | None
