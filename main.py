import json
from agent.core import Agent
from tools.calculator import CalculatorTool

def run_agent(user_input, max_turns=10):
    """
    The Agent Loop: Executes steps until a final response is generated.
    """
    # Initialize tools and agent [20]
    calculator_tool = CalculatorTool()
    agent = Agent(tools=[calculator_tool])
    i = 0
    
    current_input = user_input # Start with the initial user message

    print(f"\n--- Starting Conversation ---\n")

    while i < max_turns: # Safer than while True [20]
        i += 1
        print(f"| ITERATION {i}")
        
        # 1. Send input (User query or Tool Result) to the Agent
        response = agent.chat(current_input)
        
        # 2. Check if the model has requested a function call
        if response.function_calls:
            
            tool_results = []
            
            print(f"| > LLM Paused: Needs to use tool(s).")
            
            # 3. Process all requested tool calls
            for function_call in response.function_calls:
                tool_name = function_call.name
                tool_input = dict(function_call.args)
                
                print(f"| > Executing Tool: {tool_name} with args {tool_input}")

                # Execute the tool using the tool map [21]
                tool = agent.tool_map[tool_name]
                tool_result = tool.execute(**tool_input)
                
                # 4. Package the result for the next loop iteration [21]
                tool_results.append({
                    "type": "tool_result",
                    "tool_name": tool_name, # Needed for Gemini's tool response format
                    # The content needs to be a JSON string of the output [21]
                    "content": json.dumps(tool_result) 
                })
                print(f"| > Tool Result: {tool_result}")

            # 5. The tool results become the new 'user_input' for the next chat call [21]
            current_input = tool_results
            
        else:
            # 6. If no tool calls, the model provides the final text answer [22]
            print(f"| > Final Answer: {response.text.strip()}")
            return response.text
        
    return "Agent reached maximum execution turns."

# --- Testing the Agent ---

if __name__ == '__main__':
    
    # Test 1: Complex math requiring tool use (multi-turn logic example [23])
    print("\n\n=============== TEST CASE 1: COMPLEX MATH ===============")
    run_agent("What is 157.09 multiplied by 493.89?")
    
    # Test 2: Multi-step reasoning (demonstrates multiple loops [24])
    print("\n\n=============== TEST CASE 2: MULTI-STEP LOGIC ===============")
    run_agent("If my salary is 50,000 and I get a 10% raise, then I spend 5,000 on a holiday, what is my new balance?")
