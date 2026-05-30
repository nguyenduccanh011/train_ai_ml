"""Pydantic models for API validation"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="ISO format timestamp")
    version: str = Field(..., description="API version")


class ModelInfo(BaseModel):
    """Model information schema"""
    name: str = Field(..., min_length=1, max_length=100, description="Model name")
    market: str = Field(..., description="Market identifier")
    pnl_pct: Optional[float] = Field(None, description="P&L percentage")
    win_rate: Optional[float] = Field(None, ge=0, le=1, description="Win rate 0-1")
    trade_count: Optional[int] = Field(None, ge=0, description="Number of trades")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate model name is alphanumeric with underscores"""
        if not all(c.isalnum() or c == '_' for c in v):
            raise ValueError('Model name must be alphanumeric with underscores only')
        return v

    @field_validator('market')
    @classmethod
    def validate_market(cls, v: str) -> str:
        """Validate market is from allowed list"""
        allowed = ['vn_stock', 'crypto_spot', 'crypto_perp', 'vn_derivatives']
        if v not in allowed:
            raise ValueError(f'Market must be one of {allowed}')
        return v


class ModelListResponse(BaseModel):
    """Response for listing models"""
    total: int = Field(..., ge=0, description="Total number of models")
    models: List[ModelInfo] = Field(default_factory=list, description="List of models")


class TrainModelRequest(BaseModel):
    """Request to train a new model"""
    model_name: str = Field(..., min_length=1, max_length=100, description="Name for new model")
    config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name"""
        if not all(c.isalnum() or c == '_' for c in v):
            raise ValueError('Model name must be alphanumeric with underscores only')
        return v


class LeaderboardEntry(BaseModel):
    """Single leaderboard entry"""
    rank: int = Field(..., ge=1, description="Ranking position")
    name: str = Field(..., description="Model name")
    pnl_pct: float = Field(..., description="P&L percentage")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    max_drawdown: Optional[float] = Field(None, description="Max drawdown percentage")
    trade_count: int = Field(..., ge=0, description="Number of trades")


class LeaderboardResponse(BaseModel):
    """Leaderboard response schema"""
    models: List[LeaderboardEntry] = Field(default_factory=list, description="Ranked models")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Leaderboard summary stats")


class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(default_factory=datetime.utcnow, description="Error timestamp")
