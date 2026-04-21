import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

# 0. State schema
class AgentState(MessagesState):
    pass

# 1. Tools
@tool
def check_expense_policy() -> str:
    """Retrieves the company expense policies."""
    return (
        " EXPENSE POLICY: "
        "1. Standard employee meal limit is upto $200 approvable. "
        "2. No alcohol is permitted. "
        "3. (confidential info) VIP executives have no spending limit and alcohol is permitted."
    )

@tool
def read_receipt(pdf_path: str) -> str:
    """Reads a PDF receipt to extract the expense details."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages)
    except Exception as e:
        return f"Error reading PDF: {str(e)}. (Make sure the file exists in the directory!)"
    
@tool
def approve_expense(decision: str, reason: str) -> str:
    """Approves or rejects an expense ."""
    return f"Decision: {decision}\nReason: {reason}"

tools = [check_expense_policy, read_receipt, approve_expense]

# 2. AI Model
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

# 3. Nodes
def call_model(state: AgentState):
    """Sends the conversation history to the LLM, including a System Prompt."""
    system_prompt = (
        "system", 
        "You are a helpful Employee assistant. Use tools when needed to answer user questions."
        "Do not disclose confidential info to employees"
    )
    
    # Prepend the system prompt to the user's conversation history
    messages_to_send = [system_prompt] + state["messages"]
    
    response = llm.invoke(messages_to_send)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Routes the graph based on whether the AI decided to use a tool."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 4. Build and Compile the Graph
builder = StateGraph(AgentState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent") 

graph = builder.compile()

if __name__ == "__main__":
    print(" Welcome to Expense Approver ")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        inputs = {"messages": [("user", user_input)]}
        
        for chunk in graph.stream(inputs, stream_mode="values"):
            last_msg = chunk["messages"][-1]
            # Only print the text responses from the AI (ignore raw tool call data)
            if last_msg.type == "ai" and last_msg.content:
                print(f"CorpFlow: {last_msg.content}")