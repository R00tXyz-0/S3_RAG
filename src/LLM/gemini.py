import os
from google import genai


class GeminiLLM:

    def __init__(self, model_name="gemini-3.6-flash"):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.model_name = model_name

    def generate(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response.text