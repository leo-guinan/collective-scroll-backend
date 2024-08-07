# main.py

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError,EmailStr
from typing import List, Optional
import uvicorn
from datetime import datetime, timedelta
import jwt
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt
from urllib.parse import quote_plus
from decouple import config
import logging
import requests
import random
import re
from urllib.parse import urlparse
import chromadb
from chromadb.config import Settings


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
origins = [
    "https://x.com",
    "https://twitter.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MONGO_URI = config("MONGO_URI")
SECRET_KEY = config("SECRET_KEY")
ALGORITHM = config("ALGORITHM", default="HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=30, cast=int)
REFRESH_TOKEN_EXPIRE_DAYS = config("REFRESH_TOKEN_EXPIRE_DAYS", default=30, cast=int)

# Construct the MongoDB connection string
client = AsyncIOMotorClient(MONGO_URI)
db = client.collective_scrolling

# Models
class UserInDB(BaseModel):
    email: EmailStr
    is_verified: bool = False

class Token(BaseModel):
    access_token: str
    token_type: str


class Community(BaseModel):
    name: str

class MediaItem(BaseModel):
    type: str
    url: Optional[str]
    thumbnailUrl: Optional[str]

class Post(BaseModel):
    author: str
    content: str
    timestamp: str
    url: str
    isAd: bool
    media: Optional[List[MediaItem]] = None

class PostStore(BaseModel):
    posts: List[Post]

class Token(BaseModel):
    access_token: str
    token_type: str

class EmailRequest(BaseModel):
    email: EmailStr

class VerifyLoginRequest(BaseModel):
    email: EmailStr
    code: str


# Authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# chroma_client = chromadb.HttpClient(host=config('CHROMA_HOST'), port=8000)
# posts_collection = chroma_client.get_or_create_collection(name="posts")
# default_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=config('OPENAI_API_KEY'),
#                                                                 model_name="text-embedding-3-small"
# )




                        
# collection = chroma_client.get_or_create_collection(name=config('COLLECTION_NAME'), embedding_function=default_ef)
# Add these helper functions
def extract_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
# Helper functions
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def send_verification_email(email: str, code: str):
    url = "https://app.loops.so/api/v1/transactional"
    payload = {
        "email": email,
        "transactionalId": config("LOOPS_TRANSACTIONAL_ID"),
        "dataVariables": {
            "verification_code": code
        }
    }
    headers = {
        "Authorization": f"Bearer {config('LOOPS_API_KEY')}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to send verification email")

async def get_user(email: str):
    user = await db.users.find_one({"email": email})
    if user:
        return UserInDB(**user)

async def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        user = await get_user(email)
        return user
    except JWTError:
        return None

async def authenticate_user(username: str, password: str):
    user = await get_user(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def create_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_access_token(data: dict):
    return create_token(data, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

def create_refresh_token(data: dict):
    return create_token(data, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = await db.users.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    logging.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )

@app.get("/", tags=["health"])
async def health_check():
    try:
        # Check database connection
        await db.command("ping")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "unhealthy"

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy" if db_status == "healthy" else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_status,
        }
    )

# Routes
@app.post("/request-login")
async def request_login(email_request: EmailRequest):
    email = email_request.email
    user = await db.users.find_one({"email": email})
    if not user:
        # Create a new user if they don't exist
        user = UserInDB(email=email)
        await db.users.insert_one(user.dict())

    verification_code = ''.join(random.choices('0123456789', k=6))
    await db.verification_codes.insert_one({"email": email, "code": verification_code})
    send_verification_email(email, verification_code)
    return {"message": "Verification code sent to your email."}


@app.post("/verify-login", response_model=Token)
async def verify_login(verify_request: VerifyLoginRequest):
    email = verify_request.email
    code = verify_request.code
    verification = await db.verification_codes.find_one({"email": email, "code": code})
    if not verification:
        raise HTTPException(status_code=400, detail="Invalid verification code")
    
    await db.users.update_one({"email": email}, {"$set": {"is_verified": True}})
    await db.verification_codes.delete_one({"email": email})
    
    access_token = create_access_token(data={"sub": email})
    refresh_token = create_refresh_token(data={"sub": email})
    
    # Store refresh token in database
    await db.refresh_tokens.insert_one({"email": email, "token": refresh_token})
    
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    user = await verify_token(refresh_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    # Check if refresh token exists in database
    token_exists = await db.refresh_tokens.find_one({"email": user.email, "token": refresh_token})
    if not token_exists:
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")
    
    new_access_token = create_access_token(data={"sub": user.email})
    new_refresh_token = create_refresh_token(data={"sub": user.email})
    
    # Replace old refresh token with new one
    await db.refresh_tokens.update_one(
        {"email": user.email, "token": refresh_token},
        {"$set": {"token": new_refresh_token}}
    )
    
    return {"access_token": new_access_token, "refresh_token": new_refresh_token, "token_type": "bearer"}

@app.post("/community/join")
async def join_community(community: Community, current_user: dict = Depends(get_current_user)):
    await db.user_communities.update_one(
        {"username": current_user["username"]},
        {"$addToSet": {"communities": community.name}},
        upsert=True
    )
    return {"message": f"Joined community: {community.name}"}

@app.post("/posts")
async def store_posts(post_store: PostStore, current_user: dict = Depends(get_current_user)):
    try:
    

        stored_count = 0
        duplicate_count = 0
        ad_count = 0

        # Store posts for each community the user is part of
        for post in post_store.posts:
           
            # Check if this tweet already exists for this community
            existing_tweet = await db.posts.find_one({
                "url": post.url
            })

            if not existing_tweet:
                await db.posts.insert_one({
                    "author": post.author,
                    "content": post.content,
                    "timestamp": post.timestamp,
                    "url": post.url,
                    "isAd": post.isAd,
                    "media": [media.dict() for media in (post.media or [])],
                    "scraped_by": current_user["email"],
                    "scraped_at": datetime.utcnow()
                })
                stored_count += 1
            else:
                duplicate_count += 1
            if post.isAd:
                ad_count += 1
        return {
            "message": f"Processed {len(post_store.posts)} posts",
            "stored": stored_count,
            "duplicates": duplicate_count,
            "ads": ad_count
        }
    except Exception as e:
        logging.error(f"Error processing posts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing posts: {str(e)}")

@app.get("/search")
async def search_posts(query: str, community: str, has_media: bool = False, current_user: dict = Depends(get_current_user)):
    # Check if user is in the community
    user_community = await db.user_communities.find_one({
        "username": current_user["username"],
        "communities": community
    })
    if not user_community:
        raise HTTPException(status_code=403, detail="User not in this community")

    # Construct the search query
    search_query = {
        "community": community,
        "$text": {"$search": query}
    }
    
    if has_media:
        search_query["media"] = {"$exists": True, "$ne": []}

    # Perform search
    cursor = db.posts.find(search_query).sort("timestamp", -1).limit(20)
    
    results = await cursor.to_list(length=20)
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config("PORT", default=8000, cast=int))