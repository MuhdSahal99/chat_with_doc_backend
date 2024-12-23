from mistralai import Mistral
from app.config import settings
from typing import List

class ChatModel:
    def __init__(self):
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.model = settings.MODEL_NAME

    def get_response(self, context: List[str], question: str) -> str:
        prompt = f"""
        Answer the question based on the following context. In your response, please include references to where the information came from by mentioning relevant page numbers, headings, or subheadings that appear in the text. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."
        
        Context:
        {' '.join(context)}
        
        Question: {question}

        Please structure your response in this format:
        Answer: [Your detailed answer]
        Source references: [Specify which parts of the document (page numbers/headings/subheadings) this information came from]
        """
        
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )

        return response.choices[0].message.content