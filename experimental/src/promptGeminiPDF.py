from google.genai import types
import google.genai as genai
import pathlib

def promptGeminiPDF(pdfPath, outputPath, geminiModel, prompt, googleApiKey):
    """
    Call Gemini API then supply PDF and prompt.
    Args:
        pdfPath (str): The PDF path that will be supplied to Gemini.
        outputPath (str): The output path of the Gemini API Call.
        prompt (str): The prompt that will be supplied to the Gemini API call.
    Returns:

    """
    try:
        # Initialize Gemini Client and pass API key directly
        client = genai.Client(api_key = googleApiKey)
        filepath = pathlib.Path(pdfPath)

        # Specify the model, PDF, and Prompt that be sent using the API
        # Save the response. 
        response = client.models.generate_content(
            model= geminiModel,
            contents=[
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type='application/pdf',
                ),
                prompt])
        
        with open(outputPath, 'w', encoding='utf-8') as output_file:
                output_file.write(response.text)
        
        return f"Gemini API output saved as plain text to: {outputPath}"

    except KeyError:
        return "Error: GOOGLE_API_KEY not found in keyring. Please ensure " \
            "you have stored your Gemini API key in keyring."
    except Exception as e:
        return f"Error communicating with Gemini API or processing PDF: {e}"


if __name__ == "__main__":
    pdfInput = input("Enter PDF file input path: ")
    geminiOutput = input("Enter output path for the Gemini results: ")
    geminiModel = "gemini-2.0-flash-lite-preview-02-05"
    googleApiKey = input("Enter Google API Key: ")
    prompt = input("Enter the LLM Prompt: ")

    
    promptGeminiPDF(pdfInput, geminiOutput, geminiModel, prompt, googleApiKey)
    