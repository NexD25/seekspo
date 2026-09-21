from pydantic import BaseModel

class UserCreate(BaseModel):
    email: str
    password: str

class UserOut(BaseModel):
    id: int
    email: str

    class Config:
        from_attributes = True

class ReviewCreate(BaseModel):
    business_id: int
    rating: int
    review_text: str

class ReviewOut(BaseModel):
    id: int
    user_id: int
    business_id: int
    rating: int
    review_text: str
    sentiment: str | None = None
    aspects: str | None = None

    class Config:
        from_attributes = True

class BusinessCreate(BaseModel):
    name: str
    category: str
    area: str
    description: str

class BusinessOut(BaseModel):
    id: int
    owner_id: int | None
    name: str
    category: str
    area: str
    description: str

    class Config:
        from_attributes = True