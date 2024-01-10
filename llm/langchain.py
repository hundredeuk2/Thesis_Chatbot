from .base import LLM
from typing import Optional
import os
# from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.callbacks import get_openai_callback

from ..helpers.post_processing import remove_prefix

from dotenv import load_dotenv
from ..exceptions import APIKeyNotFoundError

load_dotenv()

class LangChain(LLM):
    """LangChain LLM"""

    def __init__(
        self,
        api_token: Optional[str] = None,
        instruction: Optional[str] = None,
        callbacks=None,
        verbose=False,
    ):
        """
        Initialize the LangChain LLM.

        Args:
            api_token (Optional[str], optional): The API token for OpenAI. If not
                provided, it will look for the "OPENAI_API_KEY" environment variable.
                Defaults to None.
            instruction (Optional[str], optional): Instruction for the OpenAI model.
                Defaults to None.
            callbacks (optional): Callback functions to be used with the OpenAI model.
                If provided, streaming will be set to True. Model will support streaming,
                where results are sent incrementally. But the token count(verbose) will
                be impossible. Defaults to None.
            verbose (bool, optional): If True, will print verbose outputs for debugging.
                Defaults to False.

        Raises:
            APIKeyNotFoundError: Raised when no OpenAI API key is provided and it's not
                found in environment variables."""

        self.api_token = api_token or os.getenv("GOOGLE_API_KEY") or None

        if self.api_token is None:
            raise APIKeyNotFoundError("OpenAI API key is required")

        self.instruction = instruction

        self.callbacks = callbacks

        self.verbose = verbose

        # callbacks이 인자로 들어오면 streaming = True로 설정
        self.streaming = False if self.callbacks is None else True

        self.llm = ChatGoogleGenerativeAI(
            model_name="gemini-pro",
            # temperature=0.7,
            streaming=self.streaming,
            convert_system_message_to_human=True,
            callbacks=callbacks,
        )  # gpt-3.5-turbo, gpt-4

        self.system_message = f"{self.instruction} 다음 논문을 참고하여 자세하게 답변해주는 AI입니다."

        self.history_memory = [SystemMessage(content=self.system_message)]

        self.initial_history_memory = [SystemMessage(content=self.system_message)]

    def call(self, prompt: str, document: str) -> str:
        with get_openai_callback() as cb:
            response = self.llm(
                self.history_memory
                + [
                    HumanMessage(
                        content=f"{prompt}\n\n### 논문:\n{document}\n"
                    )
                ]
            )

        if self.verbose:
            print(cb)

        response = response.content

        print(response)

        # 후처리
        response = remove_prefix(response)

        self.history_memory.append(HumanMessage(content=prompt))
        self.history_memory.append(AIMessage(content=response))

        if self.verbose:
            print(self.history_memory)

        return response

    def set_callbacks(self, callbacks):
        """Re-initialize the callbacks for the LLM."""
        self.llm = ChatGoogleGenerativeAI(
            # model_name="gpt-4",
            model_name="gemini-pro",
            # temperature=0.7,
            streaming=self.streaming,
            callbacks=callbacks,
        )