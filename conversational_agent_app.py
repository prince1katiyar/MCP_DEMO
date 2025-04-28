# conversational_agent_app.py - Build a full LangChain-powered conversational agent with tools and memory

import os
import openai
import datetime
import requests
import wikipedia
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain.tools import tool

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
import panel as pn
import param
from langchain_community.chat_models import ChatOpenAI

# Load API Key
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]


# ========== Define Tools ==========
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> str:
    """Fetch current temperature using Open-Meteo API for given coordinates."""
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "forecast_days": 1
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        return "Weather API failed."
    data = response.json()
    now = datetime.datetime.utcnow()
    times = [datetime.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in data['hourly']['time']]
    temps = data['hourly']['temperature_2m']
    index = min(range(len(times)), key=lambda i: abs(times[i] - now))
    return f"The current temperature is {temps[index]}°C"

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a query and return top summaries."""
    titles = wikipedia.search(query)
    summaries = []
    for title in titles[:3]:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            summaries.append(f"**{title}**\n{page.summary}")
        except:
            pass
    return "\n\n".join(summaries) if summaries else "No good result found."

@tool
def create_your_own(query: str) -> str:
    """Reverse the input text as a custom example tool."""
    return f"You sent: {query}. This reverses it: {query[::-1]}"

# ========== Register Tools ==========
tools = [get_current_temperature, search_wikipedia, create_your_own]

# ========== Panel Chatbot UI ==========
pn.extension()

class ConversationalBot(param.Parameterized):
    def __init__(self, tools, **params):
        super().__init__(**params)
        self.panels = []
        self.tool_funcs = [format_tool_to_openai_function(t) for t in tools]
        self.llm = ChatOpenAI(temperature=0).bind(functions=self.tool_funcs)
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are helpful but sassy assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        self.chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | self.prompt | self.llm | OpenAIFunctionsAgentOutputParser()

        self.executor = AgentExecutor(agent=self.chain, tools=tools, verbose=False, memory=self.memory)

    def interact(self, query):
        if not query:
            return
        result = self.executor.invoke({"input": query})
        self.answer = result['output']
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=500)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=500, styles={"background-color": "#f0f0f0"}))
        ])
        return pn.WidgetBox(*self.panels, scroll=True)


# ========== Launch the Panel Chat App ==========
cb = ConversationalBot(tools)
inp = pn.widgets.TextInput(placeholder='Ask me anything...')
conversation = pn.bind(cb.interact, inp)

tab = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True, height=400),
    pn.layout.Divider()
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# 🧠 Conversational Agent Bot')),
    pn.Tabs(('Chat', tab))
)

dashboard.servable()