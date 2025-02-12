from keyrings.alt.file import PlaintextKeyring
from google.genai import types
import google.genai as genai
import keyring
import pathlib
import PyPDF2
import os

def createSmallerPdf(pdfPath, outputPath, numPages):
    """
    Creates a new PDF containing the first 'numPages' pages of the input PDF.
    Args:
        pdfPath (str): The path to the PDF that will be read.
        outputPath (str): The path where the created PDF will be written.
        numPages (int): The number of pages from 0 to keep.
    Returns:
        str: PDF creation confirmation and path information.
    """
    try:
        # Read Original PDF and Init PDF object
        with open(pdfPath, 'rb') as pdfFile:
            pdfReader = PyPDF2.PdfReader(pdfFile)
            pdfWriter = PyPDF2.PdfWriter() 
            numPagesToCopy = min(numPages, len(pdfReader.pages))

            # Add each page within numPagesToCopy to the pdf object
            for pageNum in range(numPagesToCopy):
                page = pdfReader.pages[pageNum]
                pdfWriter.add_page(page)
            
            # Write the pdf to outputPath
            with open(outputPath, 'wb') as outputFile:
                pdfWriter.write(outputFile)

            # Return confirmation and path info
            return f"New PDF with the first {numPagesToCopy} pages created " \
                f"and saved to: {outputPath}"
    
    # Error Catch
    except FileNotFoundError:
        return f"Error: Input PDF file not found at '{pdfPath}'"
    except Exception as e:
        return f"An error occurred during PDF creation: {e}"


def promptGeminiPDF(pdfPath, outputPath, geminiModel, prompt):
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
        googleAPIKey = keyring.get_password("GradBoxLLM","GOOGLE_API_KEY")
        client = genai.Client(api_key = googleAPIKey)
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
    originalPdfInput = input("Enter input path to original PDF file: ")
    smallerPdfOutput = input("Enter output path for the smaller PDF file: ")
    numPages = 20
    geminiOutput = input("Enter output path for the Gemini results: ")
    geminiModel = "gemini-2.0-flash-lite-preview-02-05"
    prompt = input("Enter the LLM Prompt: ")

    createSmallerPdf(originalPdfInput, smallerPdfOutput, numPages)
    promptGeminiPDF(smallerPdfOutput, geminiOutput, geminiModel, prompt)
    