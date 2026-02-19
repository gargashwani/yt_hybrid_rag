from sqlalchemy import create_engine, Column, Integer, Text, JSON, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://ashwanigarg:Test123@localhost/foundry_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit = False, autoflush= False, bind=engine)
Base = declarative_base()

class FoundryAgent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, unique=True, index=True)
    agent_name = Column(String)
    agent_model = Column(String)

class UserChatSession(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, unique=True, index=True)
    foundry_conversation_id = Column(String)

Base.metadata.create_all(bind=engine)    