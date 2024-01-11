from mother import Embedding
import os
import google.generativeai as genai

from dotenv import load_dotenv
from typing import Optional
from exceptions import APIKeyNotFoundError




class GeminiEmbedding(Embedding):
    """Open Ai embedding class to implement a Embedding."""

    def __init__(self, api_token: Optional[str] = None):
        load_dotenv()
        self.api_token = api_token or os.getenv('GOOGLE_API_KEY') or None

        if self.api_token is None:
            raise APIKeyNotFoundError("OpenAI API key is required")

        genai.configure(api_key=self.api_token)

        self.model = "models/embedding-001"  # 1536 size vector

    def get_embedding(self, prompt):
        """_summary_

        Args:
            prompt (str): text for embedding

        Returns:
            vector: sentence_vector
        """
        prompt = prompt.replace("\n", " ")
        sentence_vector = genai.embed_content(
                                    model=self.model,
                                    content=[prompt],
                                    task_type="retrieval_document",
                                    title="Embedding of single string"
                                    )['embedding']
        # openai.Embedding.create(input=[prompt], model=self.model)[
        #     "data"
        # ][0]["embedding"]

        return sentence_vector