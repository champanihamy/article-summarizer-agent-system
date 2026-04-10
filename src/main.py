import os
import json
import logging
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Callable

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
MODEL = "qwen/qwen3.5-9b"
MAX_STEPS = 5

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Tool function for fetching web content
def fetch_web_content(url: str) -> str:
    """Scrapes text from a URL, cleaning up HTML boilerplate."""
    try:
        logger.info(f"Fetching content from {url}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove common noise elements
        for noise in soup(["script", "style", "nav", "footer", "aside"]):
            noise.decompose()

        # Try to find the semantic main content area
        content_area = soup.find('article') or soup.find('main') or soup.find(id='content') or soup.body
        
        if content_area:
            text = content_area.get_text(separator=' ')
        else:
            text = soup.get_text(separator=' ')
            
        # Clean whitespace
        content = " ".join(text.split())[:15000]
        
        # Log a snippet for debugging (observability)
        logger.info(f"Fetched content snippet: {content[:150]}...")
        
        return content
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return f"Error: Unable to fetch content from {url}. Technical details: {str(e)}"

# Article summarizer agent
def run_article_summarizer():
    user_url = input("Enter the article URL: ")
    
    available_functions: Dict[str, Callable] = {
        "fetch_web_content": fetch_web_content,
    }

    # Define tools for the model
    tools = [{
        "type": "function",
        "function": {
            "name": "fetch_web_content",
            "description": "Scrape and return the text content of a website.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                },
                "required": ["url"],
            },
        }
    }]

    # Initial interaction
    messages: List[Dict[str, Any]] = [
        {
            "role": "system", 
            "content": (
                "You are a helpful assistant that specializes in summarizing web articles accurately and concisely. "
                "CRITICAL: The summary MUST be written in the SAME LANGUAGE as the primary content of the article. "
                "Do not translate the summary into another language."
            )
        },
        {"role": "user", "content": f"Summarize this article: {user_url}"}
    ]

    step = 0
    while step < MAX_STEPS:
        step += 1
        
        # Prepare API call arguments
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "tools": tools,
        }
        
        if step == 1:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": "fetch_web_content"}}

        response = client.chat.completions.create(**kwargs)
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        # Check if model wants to call tools
        if not assistant_message.tool_calls:
            # Print the final result if there are no more tool calls
            print("\n--- SUMMARY ---\n", assistant_message.content)
            break

        # Execute requested tools
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions.get(function_name)

            if function_to_call:
                logger.info(f"Agent is calling tool: {function_name}")
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    result = function_to_call(**function_args)
                except Exception as e:
                    result = f"Error executing tool: {e}"
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": result,
                })
            else:
                logger.warning(f"Model requested unknown tool: {function_name}")
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error: Tool '{function_name}' not recognized.",
                })


if __name__ == "__main__":
    run_article_summarizer()