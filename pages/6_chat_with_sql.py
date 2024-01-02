import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
import os

os.environ["OPENAI_API_KEY"]    = "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"
os.environ["GOOGLE_API_KEY"]    = "AIzaSyC_GULKOkPFTFTI3GqSbd6v1U07LUJ4Ook"
os.environ["GOOGLE_CSE_ID"]     = "334c6e9dd68da48e6"
os.environ["DB_URI"]            = "mssql+pymssql://wonikadmin:wonikqnc%406139@wiq-qms-sql.database.windows.net:1433/WIQ-QMS-PROD-DB"

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with WONIK DB")

# User inputs
db_uri = os.getenv("DB_URI")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check user inputs
if not db_uri:
    st.info("Please enter database URI to connect to your database.")
    st.stop()

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Setup LLM
llm = ChatOpenAI(
    model_name="gpt-4", openai_api_key=openai_api_key, temperature=0, streaming=True
)


@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri, include_tables=[ "TB_KPI_DASHBOARD_TREND_WEEKLY" ])


db = configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=False,
    memory=conversational_memory
)

from langchain.utilities import GoogleSearchAPIWrapper
 
search = GoogleSearchAPIWrapper()
 
tools = [
     Tool(
        name='Knowledge Base',
        func=sql_agent.run,
        description=(
            'use this tool when answering general knowledge queries to get '#tool description 수정 필요
            'more information about the sqlserver'
        )
    ),
    Tool(
        name="Google Search",
        func=search.run,
        description="Search Google for recent results.",#tool description 수정 필요
     )
]

from langchain.memory.chat_message_histories import StreamlitChatMessageHistory 
from langchain.memory import ConversationBufferMemory
# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True) 
 
# Initialize agent
mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,memory=memory,handle_parsing_errors=True)
 

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = mrkl.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)