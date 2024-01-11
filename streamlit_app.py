from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType


import streamlit as st
import pandas as pd
import numpy as np
import os

from rag import Llmvdb
from rag.embeddings.gemini import GeminiEmbedding
from rag.llm.langchain import LangChain

# Preprocess
def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)
        

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

    
# Initialize your_llm only once
if "your_llm" not in st.session_state:
    stream_handler = StreamHandler(st.empty())

    embedding = GeminiEmbedding()
    llm = LangChain(
        instruction='당신은 지금부터 논문에 대해 자세히 알려주는 AI 비서입니다. 질문에 알맞는 정보를 사실대로만 대답해주세요. 대답 시 주의사항은 문서의 reference를 반드시 제공하면서 출처를 명확히 밝혀야합니다. 이 때, 출처는 모든 대답 끝에 "\n\n"을 붙여서 생성해주세요.',
        callbacks=[stream_handler],
    )
    st.session_state["your_llm"] = Llmvdb(
        embedding,
        llm,
        file_path="data/generated_data.json",
        workspace="workspace_path",
        verbose=False,
    )
    if not os.path.exists("./workspace_path/InMemoryExactNNIndexer[ToyDoc][ToyDocWithMatchesAndScores]/index.bin"):
        st.session_state["your_llm"].initialize_db()
        
    # st.session_state["your_llm"].initialize_db()

def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 2. 디렉토리가 있으니, 파일 저장
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content='안녕하세요! AI 연구 어시스턴트 스콜라 요한슨입니다. 무엇을 도와드릴까요?'
        )
    ]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# File uploader
file_name = st.sidebar.file_uploader("논문이나 엑셀파일을 업로드 하세요.", type=["pdf","xlsx"])
prompt = st.chat_input()
        
if prompt :
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = st.session_state["your_llm"].llm.set_callbacks([stream_handler])    
        
        if file_name:
            # if "xlsx" in file_name.name:
            save_uploaded_file('df_data', file_name)
            sample_df = pd.read_excel('df_data/'+ file_name.name,engine="openpyxl")
            
            # DataFrame pre-processing
            idx = [0]
            cnt = 0
            for i in range(len(sample_df)):
                for j in range(2):
                    tmp = sample_df.iloc[i,j]

                    if type(tmp) == str :
                        if  "Unnamed" in tmp:

                            cnt += 1
                            idx.append(cnt)
                            break
                    elif type(tmp) == float:
                        if np.isnan(tmp):

                            cnt += 1
                            idx.append(cnt)
                            break
            idx.append(cnt+1)
            cols = None
            if "Date" in sample_df.iloc[idx[-2],0] :
                cols = 0
            df = pd.read_excel('df_data/'+ file_name.name,engine="openpyxl", header=idx, index_col=cols)
        

            # Pandas Agent 
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.0, streaming=True), df,verbose=True,
                                agent_type=AgentType.OPENAI_FUNCTIONS,
                                handle_parsing_errors=True,
                                callbacks = [stream_handler]
                )
            
            response = agent.run(f"표를 보고 다음 질문에 대답을 해주세요. 대답을 한 후 반드시 대답을 위해 참고한 데이터프레임의 결과도 '표'형태로 대답해주세요. df 데이터프레임은 {file_name.name}에 관한 데이터입니다. 질문:"+prompt)
            st.session_state["messages"].append(
                    ChatMessage(role="assistant", content=response)
                )
            st.write(response)

            
        else:
            response = st.session_state["your_llm"].generate_response("질문:"+prompt)
            st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response)
        )
            

# streamlit run demo.py