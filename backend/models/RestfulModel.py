from typing import List, Union
from pydantic import BaseModel

class RestfulModel(BaseModel):
    resultcode : int = 200 # Response code
    message: str = None # Response message
    data: Union[List, str] = []  # Data