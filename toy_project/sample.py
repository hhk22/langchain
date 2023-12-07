
from fastapi import FastAPI
from pydantic import BaseModel, Field


class Name(BaseModel):
    name: str = Field(description="Name")
    year: int = Field(description="The year I born")


app = FastAPI(
    title="Data Analyst",
    description="I am a 6 years of Data Analyst",
    servers=[
        {"url": "https://perfume-warm-zambia-tahoe.trycloudflare.com"}
    ]
)


@app.get(
    "/name",
    summary="get my name",
    description="introduces my name",
    response_description="name(string)",
    response_model=Name
)
def get_analyze():
    return {"name": "Hyeonghwan", "year": 1992}
