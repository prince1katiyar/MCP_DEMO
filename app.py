import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# If using OpenAI instead of Groq, uncomment below and comment above
# from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
import sys

# Define an asynchronous function for the chat loop
async def run_memory_chat():
    """Run a chat using MCPAgent with built-in conversation memory."""

    # --- 1. Load Environment Variables ---
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found in .env file.")
        sys.exit(1)

    # --- 2. Set Configuration File Path ---
    # Ensure this file exists in the same directory or provide the full path
    config_file = "browser_mcp.json"
    print(f"Using configuration file: {config_file}")
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found at {config_file}")
        sys.exit(1)

    print("Initializing chat...")

    client = None # Initialize client to None for cleanup check
    try:
        # --- 3. Create MCP Client and Initialize LLM ---
        # Create an MCP client instance from the configuration file
        # CORRECTED LINE: Pass the config_file variable directly
        client = MCPClient.from_config_file(config_file)

        # Initialize the Large Language Model (LLM) - using Groq here
        llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
        # llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY")) # Example for OpenAI

        # --- 4. Create the MCPAgent ---
        agent = MCPAgent(
            llm=llm,
            client=client,
            max_steps=15,
            memory_enabled=True,
        )

        # --- 5. Start Interactive Chat Loop ---
        print("\n======== Interactive MCP Chat ========")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("Type 'clear' to clear conversation history.")
        print("=======================================\n")

        while True:
            try:
                 user_input = input("\nYou: ")
            except EOFError:
                 break

            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break

            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("(Conversation history cleared.)")
                continue

            print("\nAssistant: ", end="", flush=True)
            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                 # Print more detailed error if possible
                 import traceback
                 print(f"\nError during agent execution: {e}")
                 # traceback.print_exc() # Uncomment for full traceback if needed

    except KeyboardInterrupt:
         print("\nExiting chat due to interrupt...")
    except Exception as e:
         print(f"\nAn error occurred during initialization or chat loop: {e}")
         import traceback
         traceback.print_exc() # Print initialization errors
    finally:
        # --- 6. Clean Up ---
        print("\nCleaning up MCP client sessions...")
        if client and client.sessions:
            try:
                await client.close_all_sessions()
                print("MCP sessions closed.")
            except Exception as e:
                print(f"Error closing MCP sessions: {e}")
        else:
             print("No active MCP sessions to clean up or client not initialized.")
        print("Cleanup complete. Goodbye!")

# --- Main Execution Block ---
if __name__ == "__main__":
    asyncio.run(run_memory_chat())