import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from enum import Enum
from pydantic import BaseModel
from typing import Tuple

# Load environment variables from .env file
load_dotenv()

class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""
    GEMINI = "Gemini"


class LLMModel(BaseModel):
    """Represents an LLM model configuration"""
    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        # Gemini models generally support structured output
        return True


# Define available models
AVAILABLE_MODELS = [
    LLMModel(
        display_name="[gemini] gemini-pro",
        model_name="gemini-pro",
        provider=ModelProvider.GEMINI
    )
]

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

def get_model_info(model_name: str) -> LLMModel | None:
    """Get model information by model_name"""
    return next((model for model in AVAILABLE_MODELS if model.model_name == model_name), None)

def get_model(model_name: str, model_provider: ModelProvider) -> ChatGoogleGenerativeAI | None:
    """Fetch the model instance based on provider"""
    if model_provider == ModelProvider.GEMINI:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure GOOGLE_API_KEY is set in your .env file.")
            raise ValueError("Google API key not found. Please make sure GOOGLE_API_KEY is set in your .env file.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    return None  # Only Gemini is supported in this version