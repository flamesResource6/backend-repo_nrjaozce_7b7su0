"""
Database Schemas for VectorTutor

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercased class name (e.g., Flashcard -> "flashcard").
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class Material(BaseModel):
    user_id: str = Field(..., description="Owner of this material")
    title: str = Field(..., description="Material title")
    type: Literal["pdf", "slides", "image", "text"] = Field(...)
    text: str = Field(..., description="Raw extracted text content")
    topics: List[str] = Field(default_factory=list, description="Top-level topics segmented from content")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Flashcard(BaseModel):
    user_id: str
    material_id: str
    topic: str
    question: str
    answer: str
    tags: List[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    source_refs: List[str] = Field(default_factory=list)

class QuizItem(BaseModel):
    user_id: str
    material_id: str
    topic: str
    question: str
    options: List[str]
    correct_index: int
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    explanation: Optional[str] = None

class Plan(BaseModel):
    user_id: str
    schedule: List[Dict[str, Any]] = Field(default_factory=list, description="List of study sessions with topics and due dates")
    goals: List[str] = Field(default_factory=list)

class Performance(BaseModel):
    user_id: str
    material_id: Optional[str] = None
    topic: Optional[str] = None
    accuracy: float = 0.0
    attempts: int = 0
    streak: int = 0
    last_activity: Optional[str] = None

class ChatHistory(BaseModel):
    user_id: str
    material_id: Optional[str] = None
    question: str
    answer: str
    refs: List[str] = Field(default_factory=list)
