from pydantic import BaseModel

class ResultInfoModel(BaseModel):
    FileName:str="test"
    FileId:str="test"
    TB_InferenceResult:bool = False 
