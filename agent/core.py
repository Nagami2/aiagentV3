import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class Agent:
    """
    The core AI Agent class managing conversation state, LLM interaction,
    and tool schema exposure.
    """

    def __init__(self, tools=None, model="gemini-2.5-flash"):
        # Component 1: LLM Setup
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model
        
        # Component 1: Instructions (System Prompt)
        self.system_message = "You are a helpful assistant that breaks down problems into steps and solves them systematically. Use the provided tools when necessary."
        
        # Component 2: Conversation Memory (History) [11, 12]
        # Stores history as a list of types.Content objects
        self.messages = []
        
        # Tool handling setup [13]
        self.tools = tools if tools is not None else []
        self.tool_map = {tool.get_schema().name: tool for tool in self.tools}

    def _get_tool_schemas(self):
        """Extracts tool schemas for the API call."""
        if not self.tools:
            return None
        
        # Gemini requires tools to be wrapped in a list of Tool objects [9]
        function_declarations = [tool.get_schema() for tool in self.tools]
        return [types.Tool(function_declarations=function_declarations)]

    def _prepare_contents(self, current_input):
        """
        Prepares the list of contents (history + current input) to send to the LLM.
        This handles both user messages (string) and tool results (list of tool response dicts).
        """
        contents = self.messages[:]

        if isinstance(current_input, str):
            # If it's a new user message
            user_content = types.Content(
                role='user',
                parts=[types.Part.from_text(text=current_input)]
            )
            contents.append(user_content)
            
        elif isinstance(current_input, list) and current_input and current_input.get('type') == 'tool_result':
            # If it's a tool result coming back from the execution loop [14]
            tool_response_parts = []
            for result_block in current_input:
                # Format the tool result into a types.Part.from_function_response [15]
                function_response_part = types.Part.from_function_response(
                    name=result_block['tool_name'],
                    response=json.loads(result_block['content'])
                )
                tool_response_parts.append(function_response_part)
            
            # Tool results must be sent back with the 'tool' role [15]
            tool_content = types.Content(
                role='tool',
                parts=tool_response_parts
            )
            contents.append(tool_content)
        
        return contents

    def chat(self, message):
        """
        Processes an input message (user query or tool result) and sends it to the LLM.
        """
        contents = self._prepare_contents(message)

        # 1. Configuration (System Instructions and Tools) [16]
        config = types.GenerateContentConfig(
            system_instruction=self.system_message,
            max_output_tokens=1024,
            temperature=0.1,
            tools=self._get_tool_schemas()
        )

        # 2. API Call
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents, # Full conversation history is passed [17]
            config=config,
        )

        # 3. Store the model's response for memory (regardless of whether it's text or a tool call)
        # Note: We must update self.messages after the API call [17]
        if response.function_calls:
            # If model requests a tool, store the tool call instruction (role: 'model') [18]
            model_parts = [types.Part.from_function_call(fc) for fc in response.function_calls]
            self.messages.append(types.Content(role='model', parts=model_parts))
        
        elif response.text:
            # If model returns text, store the response (role: 'model')
            self.messages.append(types.Content(role='model', parts=[types.Part.from_text(text=response.text)]))

        return response
