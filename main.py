from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Use this tool for mathematical calculations. Input should be a math expression like '2+3' or '10*5'"""
    try:
        result = eval(expression)
        return f"Calculation: {expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@tool
def get_current_time() -> str:
    """Use this tool to get the current date and time"""
    from datetime import datetime
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def word_count(text: str) -> str:
    """Use this tool to count words in a given text"""
    words = text.split()
    return f"Word count: '{text}' has {len(words)} words"

@tool
def text_reverse(text: str) -> str:
    """Use this tool to reverse a given text"""
    return f"Reversed: '{text}' → '{text[::-1]}'"

@tool
def weather_info(city: str) -> str:
    """Use this tool to get weather information for a specific city"""
    return f"Weather in {city}: Sunny, 22°C (Mock data)"

def main():
    model = ChatOpenAI(temperature=0)

    tools = [calculator, get_current_time, word_count, text_reverse, weather_info]
    agent = create_react_agent(model, tools)

    print("Welcome! I'm your AI assistant with the following tools:")
    print("🧮 Calculator - perform math operations")
    print("⏰ Time - get current date and time")
    print("📝 Word Count - count words in text")
    print("🔄 Text Reverse - reverse any text")
    print("🌤️  Weather - get weather info (mock data)")
    print("Type 'exit' to end the conversation")

    while True:
        user_input = input("You: ").strip()

        if user_input == "exit":
            break

        print("\nAssistant:", end="")
        try:
            response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            print(response["messages"][-1].content)
            
            # Debug: Show tool usage
            tool_calls = 0
            for msg in response["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls += len(msg.tool_calls)
            
            if tool_calls > 0:
                print(f"\n[DEBUG: Used {tool_calls} tool(s)]")
            else:
                print(f"\n[DEBUG: No tools used - direct response]")
                
        except Exception as e:
            print(f"Error: {e}")
        print()

if __name__ == "__main__":
    main()