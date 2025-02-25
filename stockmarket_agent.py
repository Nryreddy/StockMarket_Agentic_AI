import openai
from phi.agent import Agent
from phi.model.groq import Groq    # model inference
from phi.tools.yfinance import YFinanceTools   #YFinanceTools is a tool that provides stock price, analyst recommendations, stock fundamentals, and company news
from phi.tools.duckduckgo import DuckDuckGo

import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")

## web search agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for latest the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Include the sources every time you provide information"],
    show_tools_calls=True,
    markdown=True,

)

## Stock Market agent
finance_agent=Agent(
    name="Stock Market AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(stock_price=True,company_info=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True,historical_prices=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    instructions=["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA",stream=True)
