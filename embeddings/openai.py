from mother import Embedding
import os
import openai
from dotenv import load_dotenv
from typing import Optional
from exceptions import APIKeyNotFoundError




class OpenAIEmbedding(Embedding):
    """Open Ai embedding class to implement a Embedding."""

    def __init__(self, api_token: Optional[str] = None):
        load_dotenv()
        self.api_token = api_token or os.getenv("OPENAI_API_KEY") or None

        if self.api_token is None:
            raise APIKeyNotFoundError("OpenAI API key is required")

        openai.api_key = self.api_token

        self.model = "text-embedding-ada-002"  # 1536 size vector

    def get_embedding(self, prompt):
        """_summary_

        Args:
            prompt (str): text for embedding

        Returns:
            vector: sentence_vector
        """
        prompt = prompt.replace("\n", " ")
        sentence_vector = openai.Embedding.create(input=[prompt], model=self.model)[
            "data"
        ][0]["embedding"]

        return sentence_vector