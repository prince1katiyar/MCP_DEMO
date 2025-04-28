import asyncio
import json
import os
import nest_asyncio
import pprint
import base64
from io import BytesIO
import pandas as pd
from playwright.async_api import async_playwright
from openai import OpenAI
from PIL import Image
from tabulate import tabulate
from IPython.display import display, HTML, Markdown, Image as IPImage # Renamed to avoid clash
from pydantic import BaseModel, Field, HttpUrl # Import Field and HttpUrl for better validation
from typing import Optional, List # Import Optional and List
from dotenv import load_dotenv

load_dotenv() 

# --- Helper Function (Assuming helper.py content) ---
# It's better practice to have this in a separate file,
# but including a simple version here for completeness.
def get_openai_api_key():
    """Retrieves the OpenAI API key from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    return api_key

# --- OpenAI Client Initialization ---
try:
    client = OpenAI(api_key=get_openai_api_key())
except ValueError as e:
    print(f"Error initializing OpenAI client: {e}")
    # Handle the error appropriately, e.g., exit or prompt the user
    exit() # Exit if API key is missing

# --- Apply nest_asyncio ---
# Allows asyncio event loop to be nested, useful in environments like Jupyter
nest_asyncio.apply()

# --- Web Scraper Agent Class (Unchanged) ---
class WebScraperAgent:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        print("WebScraperAgent initialized.")

    async def init_browser(self):
      print("Initializing browser...")
      try:
          self.playwright = await async_playwright().start()
          self.browser = await self.playwright.chromium.launch(
              headless=True, # Keep headless=True for server environments
              args=[
                  "--disable-dev-shm-usage",
                  "--no-sandbox",
                  "--disable-setuid-sandbox",
                  "--disable-accelerated-2d-canvas",
                  "--disable-gpu",
                  "--no-zygote",
                  "--disable-audio-output",
                  "--disable-software-rasterizer",
                  "--disable-webgl",
                  "--disable-web-security", # Be cautious with this in untrusted environments
                  "--disable-features=LazyFrameLoading",
                  "--disable-features=IsolateOrigins",
                  "--disable-background-networking"
              ]
          )
          self.page = await self.browser.new_page()
          print("Browser initialized successfully.")
      except Exception as e:
          print(f"Error initializing browser: {e}")
          raise # Re-raise the exception to handle it upstream

    async def scrape_content(self, url):
        if not self.page or self.page.is_closed():
            print("Page not found or closed, re-initializing browser...")
            await self.init_browser()
        print(f"Navigating to {url}...")
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=60000) # Increased timeout, wait for DOM
            print("Page loaded. Waiting for dynamic content...")
            await self.page.wait_for_timeout(3000)  # Wait a bit longer for dynamic content if needed
            print("Retrieving page content...")
            content = await self.page.content()
            print("Page content retrieved.")
            return content
        except Exception as e:
            print(f"Error during scraping content from {url}: {e}")
            # Attempt screenshot for debugging even if content retrieval fails
            try:
                await self.take_screenshot("error_screenshot.png")
                print("Saved error screenshot to error_screenshot.png")
            except Exception as se:
                print(f"Could not take error screenshot: {se}")
            return None # Return None or raise error

    async def take_screenshot(self, path="screenshot.png"):
        if not self.page or self.page.is_closed():
             print("Cannot take screenshot, page is not available.")
             return None
        print(f"Taking full page screenshot: {path}")
        try:
            await self.page.screenshot(path=path, full_page=True)
            print("Screenshot saved.")
            return path
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None

    async def screenshot_buffer(self):
        if not self.page or self.page.is_closed():
             print("Cannot take screenshot buffer, page is not available.")
             return None
        print("Taking viewport screenshot buffer...")
        try:
            screenshot_bytes = await self.page.screenshot(type="png", full_page=False) # Viewport screenshot
            print("Screenshot buffer captured.")
            return screenshot_bytes
        except Exception as e:
            print(f"Error taking screenshot buffer: {e}")
            return None

    async def close(self):
        print("Closing browser and Playwright...")
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self.playwright = None
        self.browser = None
        self.page = None
        print("Browser and Playwright closed.")

# --- Pydantic Models for Euron.one Articles ---
# Define the structure of the data you want to extract from euron.one
class EuronArticle(BaseModel):
    title: str = Field(..., description="The main title of the article")
    articleUrl: Optional[HttpUrl] = Field(None, description="The direct URL to the full article") # Use HttpUrl for validation
    imageUrl: Optional[HttpUrl] = Field(None, description="The URL of the main image associated with the article")
    excerpt: Optional[str] = Field(None, description="A short summary or excerpt of the article")

class EuronArticleList(BaseModel):
    articles: List[EuronArticle] = Field(..., description="A list of articles extracted from the page")

# --- LLM Processing Function (Updated for Articles) ---
async def process_with_llm(html, instructions, truncate = False):
    if not html:
        print("No HTML content received for LLM processing.")
        return None
    print("Sending content to LLM for processing...")
    # Determine content length for potential truncation
    max_len = 150000
    content_to_send = html
    if truncate and len(html) > max_len:
        print(f"HTML content truncated to {max_len} characters.")
        content_to_send = html[:max_len]
    elif len(html) > 200000: # Add a general warning for very large HTML
         print(f"Warning: HTML content is large ({len(html)} characters), processing might be slow or hit limits.")

    try:
        completion = client.chat.completions.create( # Use the standard create method
            model="gpt-4o-mini-2024-07-18", # Or "gpt-4o" for potentially better results but higher cost
            messages=[{
                "role": "system",
                "content": f"""
                You are an expert web scraping agent. Your task is to:
                Analyze the provided HTML content and extract information based on the user's instructions.
                Return the extracted data structured as JSON, conforming *exactly* to the requested format.

                User Instructions:
                {instructions}

                Specifically, identify the main news articles or blog posts presented on the page. For each article, extract:
                1.  `title`: The main headline or title of the article.
                2.  `articleUrl`: The full URL linking to the article's page. If it's a relative URL, make it absolute based on the site's domain (euron.one).
                3.  `imageUrl`: The full URL of the primary image associated with the article.
                4.  `excerpt`: A brief summary or the first few sentences of the article, if available.

                Return ONLY the valid JSON object conforming to the EuronArticleList schema, with no introductory text, markdown formatting, or explanations.
                """
            }, {
                "role": "user",
                "content": content_to_send
            }],
            temperature=0.1,
            response_format={"type": "json_object"}, # Request JSON output
            # max_tokens=4096 # Optional: Set max tokens if needed
        )

        # Parse the JSON response string into the Pydantic model
        response_content = completion.choices[0].message.content
        print("LLM response received.")
        # print("Raw LLM response:", response_content) # Uncomment for debugging

        # Validate and parse the JSON response using the Pydantic model
        parsed_data = EuronArticleList.model_validate_json(response_content)
        print("LLM response successfully parsed and validated.")
        return parsed_data

    except json.JSONDecodeError as json_err:
        print(f"❌ Error: Failed to decode JSON from LLM response.")
        print(f"Raw response was: {response_content}")
        print(f"JSON Decode Error: {json_err}")
        return None
    except Exception as e:
        print(f"❌ Error during LLM processing: {str(e)}")
        # Consider logging the full error and response content if available
        if 'response_content' in locals():
            print(f"Raw response causing error (if available): {response_content}")
        return None


# --- Main Web Scraper Function (Updated) ---
async def webscraper(target_url, instructions):
    result = None
    screenshot_bytes = None
    scraper = WebScraperAgent() # Instantiate the agent

    try:
        # Initialize browser (moved inside try block)
        await scraper.init_browser()

        # Scrape content
        print(f"\n--- Starting scraping process for: {target_url} ---")
        print("Extracting HTML Content...")
        html_content = await scraper.scrape_content(target_url)

        if html_content:
            print("HTML content extracted successfully.")
            # Capture screenshot
            print("Taking Screenshot...")
            screenshot_bytes = await scraper.screenshot_buffer()
            if screenshot_bytes:
                print("Screenshot captured successfully.")
            else:
                print("Failed to capture screenshot.")

            # Process content with LLM
            print("Processing content with LLM...")
            result: Optional[EuronArticleList] = await process_with_llm(html_content, instructions, True) # Enable truncation

            if result:
                print("\n--- LLM processing completed successfully ---")
                print(f"Extracted {len(result.articles)} articles.")
            else:
                print("\n--- LLM processing failed or returned no data ---")
        else:
            print("\n--- Failed to extract HTML content, stopping process ---")

    except Exception as e:
        print(f"❌ An unexpected error occurred during the web scraping process: {str(e)}")
        # Optionally log the full traceback here
    finally:
        print("\n--- Closing scraper resources ---")
        await scraper.close() # Ensure resources are always closed

    return result, screenshot_bytes

# --- Execution ---
async def main():
    target_url = "https://www.bbc.com/news"  # Target the specific website
    base_url = "https://www.bbc.com/news" # Base URL for resolving relative links if needed
    instructions = f"""
    Extract the main articles displayed on the homepage '{target_url}'.
    Focus on items that look like news posts or blog entries.
    Provide the title, the full article URL, the main image URL, and a short excerpt for each.
    """

    print("--- Running Web Scraper ---")
    result, screenshot = await webscraper(target_url, instructions)
    print("\n--- Scraping Finished ---")

    # --- Display Results ---
    if screenshot:
        print("\n--- Screenshot ---")
        # Display screenshot if in an environment like Jupyter
        try:
            display(IPImage(data=screenshot)) # Use IPImage from IPython
        except NameError:
            print("Cannot display image directly (IPython environment likely not detected).")
            # Optionally save the screenshot to a file
            try:
                with open("euron_screenshot.png", "wb") as f:
                    f.write(screenshot)
                print("Screenshot saved to euron_screenshot.png")
            except Exception as e:
                print(f"Error saving screenshot: {e}")

    if result and result.articles:
        print("\n--- Extracted Articles ---")
        # Option 1: Pretty Print
        # pprint.pprint(result.model_dump()) # Use model_dump() for Pydantic v2

        # Option 2: Convert to Pandas DataFrame and display as table
        try:
            df = pd.DataFrame([article.model_dump() for article in result.articles])
            print(tabulate(df, headers='keys', tablefmt='grid'))
            # In Jupyter/IPython, you might prefer:
            # display(df)
        except Exception as e:
             print(f"Could not display results as table: {e}")
             print("Raw data:")
             pprint.pprint(result.model_dump()) # Fallback to pprint

    elif result and not result.articles:
        print("\n--- No articles extracted by the LLM ---")
    else:
        print("\n--- No results obtained ---")

# --- Run the main async function ---
if __name__ == "__main__":
    # In a standard Python script, use asyncio.run()
    # If running in Jupyter/Colab, you might just await main() directly
    # because nest_asyncio allows it.
    # Check if an event loop is already running (common in Jupyter)
    try:
        loop = asyncio.get_running_loop()
        print("Asyncio loop already running.")
        # Schedule main() to run in the existing loop
        # loop.create_task(main()) # This might not wait for completion in some environments
        # If in Jupyter/IPython, simply calling await main() at the top level works due to nest_asyncio
        # If running as .py file, the asyncio.run(main()) below is correct.
        # For broader compatibility:
        if loop.is_running():
             print("Running main() in existing loop (likely Jupyter/IPython). Use 'await main()' directly in a cell.")
             # If you MUST run from __main__ block in such env, it's tricky.
             # Often just defining main and calling `await main()` in a notebook cell is easier.
        else:
             asyncio.run(main())

    except RuntimeError: # No running event loop
        print("No running asyncio loop found, starting new one.")
        asyncio.run(main())