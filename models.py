from pydantic import BaseModel


class YCStartup(BaseModel):
    id: str
    name: str
    description: str
    tagline: str
    industry: str

