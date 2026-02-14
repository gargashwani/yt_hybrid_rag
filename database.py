from sqlalchemy import create_engine, Column, Integer, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

<<<<<<< HEAD
# sqlalchemy db url
SQLALCHEMY_DATABASE_URL = "postgresql://ashwanigarg:'Test123'@localhost/rag_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DocumentChunk(Base):
    __tablename__="document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable = False)
    metadata_info = Column(JSON)


# Create the tables
=======
DATABASE_URL = "postgresql://ashwanigarg:Test123@localhost/rag_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit = False, autoflush= False, bind=engine)
Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    metadata_info = Column(JSON)


>>>>>>> master
Base.metadata.create_all(bind=engine)    