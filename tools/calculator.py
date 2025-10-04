from google.genai import types

class CalculatorTool():
    """
    A tool for performing basic mathematical calculations.
    """

    def get_schema(self):
        """
        Returns the FunctionDeclaration schema required by the Gemini API.
        """
        # This defines the tool's structured description for the LLM [8, 9]
        return types.FunctionDeclaration(
            name='calculator',
            description='Performs basic mathematical calculations (addition, subtraction, multiplication, division). Must be used for all non-trivial math.',
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'expression': types.Schema(
                        type='STRING',
                        description='The mathematical expression to evaluate (e.g., "2+2", "10*5").'
                    )
                },
                required=['expression'],
            ),
        )

    def execute(self, expression):
        """
        Evaluate mathematical expressions. (Warning: uses eval() for simplicity) [10]
        """
        try:
            # Executes the actual function logic [8]
            result = eval(expression)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

if __name__ == '__main__':
    # Simple test of the tool execution
    tool = CalculatorTool()
    print(tool.execute("10 * 10"))