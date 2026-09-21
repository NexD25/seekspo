from fastapi import FastAPI
from database import engine, Base
from models import Business, User, Review
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException
from database import SessionLocal
from schemas import UserCreate, UserOut
from auth_utils import hash_password
from fastapi.security import OAuth2PasswordRequestForm
from auth_utils import verify_password, create_access_token
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from jose import jwt
from auth_config import SECRET_KEY, ALGORITHM
from schemas import ReviewCreate, ReviewOut, BusinessCreate, BusinessOut
from ai_utils import analyze_review


Base.metadata.create_all(bind=engine)

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.post("/signup", response_model=UserOut)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(email=user.email, hashed_password=hash_password(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.get("/")
def read_root():
    return {"message": "SeekSpo API is alive"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserOut)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/reviews", response_model=ReviewOut)
def create_review(review: ReviewCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    business = db.query(Business).filter(Business.id == review.business_id).first()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    analysis = analyze_review(review.review_text)

    new_review = Review(
        user_id=current_user.id,
        business_id=review.business_id,
        rating=review.rating,
        review_text=review.review_text,
        sentiment=analysis["sentiment"],
        aspects=",".join(analysis["aspects"]),
    )
    db.add(new_review)
    db.commit()
    db.refresh(new_review)
    return new_review

@app.post("/reviews/{review_id}/analyze", response_model=ReviewOut)
def analyze_review_endpoint(review_id: int, db: Session = Depends(get_db)):
    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    result = analyze_review(review.review_text)
    review.sentiment = result["sentiment"]
    review.aspects = ",".join(result["aspects"])

    db.commit()
    db.refresh(review)
    return review

@app.post("/businesses", response_model=BusinessOut)
def create_business(business: BusinessCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_business = Business(
        owner_id=current_user.id,
        name=business.name,
        category=business.category,
        area=business.area,
        description=business.description,
    )
    db.add(new_business)
    db.commit()
    db.refresh(new_business)
    return new_business

@app.get("/businesses", response_model=list[BusinessOut])
def list_businesses(db: Session = Depends(get_db)):
    return db.query(Business).all()

@app.get("/businesses/{business_id}", response_model=BusinessOut)
def get_business(business_id: int, db: Session = Depends(get_db)):
    business = db.query(Business).filter(Business.id == business_id).first()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")
    return business

@app.get("/businesses/{business_id}/reviews", response_model=list[ReviewOut])
def get_business_reviews(business_id: int, db: Session = Depends(get_db)):
    return db.query(Review).filter(Review.business_id == business_id).all()