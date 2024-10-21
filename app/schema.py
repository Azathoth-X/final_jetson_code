from pydantic import BaseModel

class ResultInfoModel(BaseModel):
    FileName:str
    FileId:str
    TB_InferenceResult:bool = False
