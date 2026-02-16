from sqlalchemy import create_engine, Column, Integer, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://ashwanigarg:Test123@localhost/rag_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit = False, autoflush= False, bind=engine)
Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    metadata_info = Column(JSON)


class Thread(Base):
    __tablename__ = "threads"
    id= Column(Integer, primary_key=True, index=True)
    thread_id= Column(Text, unique=True, index=True)
    user_id=Column(Integer, nullable=True)


class Agent(Base):
    __tablename__ = "agents"
    id=Column(Integer, primary_key=True, index=True)
    agent_id=Column(Text, unique=True, index=True)
    agent_name=Column(Text, nullable=True)
    agent_model=Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)    