# main.py

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime, timedelta
import jwt
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt
from urllib.parse import quote_plus
from decouple import config
import logging

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


# Construct the MongoDB connection string
client = AsyncIOMotorClient(MONGO_URI)
db = client.collective_scrolling

# Models
class UserIn(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    username: str

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
    media: List[MediaItem]


class Post(BaseModel):
    author: str
    content: str
    timestamp: str
    url: str

class PostStore(BaseModel):
    community: str
    posts: List[Post]

class Token(BaseModel):
    access_token: str
    token_type: str

# Authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_user(username: str):
    return await db.users.find_one({"username": username})

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

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = await get_user(username)
    if user is None:
        raise credentials_exception
    return user

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
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/signup", response_model=UserOut)
async def signup(user: UserIn):
    existing_user = await get_user(user.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    hashed_password = get_password_hash(user.password)
    new_user = {"username": user.username, "hashed_password": hashed_password}
    await db.users.insert_one(new_user)
    return {"username": user.username}

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
    # Get all communities the user is part of
    user_communities = await db.user_communities.find_one({"username": current_user["username"]})
    if not user_communities or not user_communities.get("communities"):
        raise HTTPException(status_code=400, detail="User is not part of any community")

    communities = user_communities["communities"]

    stored_count = 0
    duplicate_count = 0

    # Store posts for each community the user is part of
    for post in post_store.posts:
        for community in communities:
            # Check if this tweet already exists for this community
            existing_tweet = await db.posts.find_one({
                "community": community,
                "url": post.url
            })

            if not existing_tweet:
                await db.posts.insert_one({
                    "community": community,
                    "author": post.author,
                    "content": post.content,
                    "timestamp": post.timestamp,
                    "url": post.url,
                    "media": [media.dict() for media in post.media],  # Store media information
                    "scraped_by": current_user["username"],
                    "scraped_at": datetime.utcnow()
                })
                stored_count += 1
            else:
                duplicate_count += 1

    return {
        "message": f"Processed {len(post_store.posts)} posts for {len(communities)} communities",
        "stored": stored_count,
        "duplicates": duplicate_count
    }

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