import os
from cerebras.cloud.sdk import Cerebras
from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class ChatHistory:
    """Keep track of conversation history for CSM context"""
    messages: List[Dict]  # List of message dicts with role and content
    speaker_map: Dict[str, int]  # Maps role to CSM speaker_id

class CerebrasClient:
    def __init__(self, api_key: str):
        self.client = Cerebras(api_key=api_key)
        # Map roles to CSM speaker IDs (0 for AI, 1 for user)
        self.speaker_map = {
            "assistant": 0,
            "user": 1
        }
        self.conversations: Dict[str, ChatHistory] = {}

    def _init_conversation(self, context_id: str):
        """Initialize a new conversation context"""
        if context_id not in self.conversations:
            self.conversations[context_id] = ChatHistory(
                messages=[],
                speaker_map=self.speaker_map.copy()
            )

    async def get_response(self, text: str, context_id: str) -> str:
        """Get response from Cerebras model and maintain conversation history"""
        self._init_conversation(context_id)
        chat_history = self.conversations[context_id]

        # Add user message to history
        chat_history.messages.append({
            "role": "user",
            "content": text
        })

        try:
            # Get completion from Cerebras
            chat_completion = self.client.chat.completions.create(
                messages=chat_history.messages,
                # model="llama3.1-8b",  # Using Llama 3.1 8B model
                model="llama-4-scout-17b-16e-instruct",  # Using Llama 4 Scout
                temperature=0.6
            )

            # Extract response
            response = chat_completion.choices[0].message.content

            # Add assistant response to history
            chat_history.messages.append({
                "role": "assistant",
                "content": response
            })

            # Limit history to last 10 messages to prevent context overflow
            if len(chat_history.messages) > 10:
                chat_history.messages = chat_history.messages[-10:]

            return response

        except Exception as e:
            print(f"Error getting Cerebras response: {e}")
            return f"Error: Failed to get AI response - {str(e)}"

    def clear_context(self, context_id: str):
        """Clear conversation history for a given context"""
        if context_id in self.conversations:
            del self.conversations[context_id]