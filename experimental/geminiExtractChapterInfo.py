import PyPDF2
import google.genai as genai
from google.genai import types
import os
import pathlib
import keyring
from keyrings.alt.file import PlaintextKeyring


def createSmallerPdf(pdfPath, outputPath, numPages=10):
    """
    Creates a new PDF containing the first 'numPages' pages of the input PDF.
    (No changes in this function)
    """
    try:
        with open(pdfPath, 'rb') as pdfFile:
            pdfReader = PyPDF2.PdfReader(pdfFile)
            pdfWriter = PyPDF2.PdfWriter()
            numPagesToCopy = min(numPages, len(pdfReader.pages)) # Don't exceed input PDF page count

            for pageNum in range(numPagesToCopy):
                page = pdfReader.pages[pageNum]
                pdfWriter.add_page(page)

            with open(outputPath, 'wb') as outputFile:
                pdfWriter.write(outputFile)
            return f"New PDF with the first {numPagesToCopy} pages created and saved to: {outputPath}"

    except FileNotFoundError:
        return f"Error: Input PDF file not found at '{pdfPath}'"
    except Exception as e:
        return f"An error occurred during PDF creation: {e}"


def extractChapterInfoGeminiPdfDirect(tocPdfPath):
    """
    Uses Gemini API (google.genai library) to extract chapter information directly from a PDF file.

    Args:
        tocPdfPath (str): Path to the PDF file containing the Table of Contents.

    Returns:
        str: A string containing the extracted chapter information in a structured format,
             or an error message if there are issues with PDF reading or Gemini API.
    """
    try:
        # **No genai.configure() here for google.genai library**
        client = genai.Client(api_key = keyring.get_password("GradBoxLLM","GOOGLE_API_KEY")) # Initialize Gemini Client and pass API key directly

        filepath = pathlib.Path(tocPdfPath) # Create a Path object for the PDF

        prompt_text = "Extract the chapter information (chapter number, chapter name, starting page) from this Table of Contents. Return in a structured format."


        response = client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp-01-21", # Or try "gemini-pro-vision" if flash doesn't work
            contents=[
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type='application/pdf',
                ),
                prompt_text]) # Send PDF part and text prompt

        gemini_output = response.text
        return gemini_output

    except KeyError:
        return "Error: GOOGLE_API_KEY not found in keyring. Please ensure you have stored your Gemini API key in keyring."
    except Exception as e:
        return f"Error communicating with Gemini API or processing PDF: {e}"


if __name__ == "__main__":
    pdfPath = "/home/wtmartinez/Downloads/medicalSurgicalNursing-AssessmentAndManagementOfClincialProblems.pdf"  # Replace with the actual path to your PDF
    tocPdfOutputPath = "/home/wtmartinez/Downloads/medicalSurgicalNursing-AssessmentAndManagementOfClincialProblemsTOC.pdf" # Output path for the smaller TOC PDF
    chapterInfoOutputPath = "/home/wtmartinez/Downloads/medicalSurgicalNursing-AssessmentAndManagementOfClincialProblemsChapterInfo.txt" # Output for chapter info from Gemini (from PDF Direct API)
    numPagesForTOC = 10 # Number of pages for the smaller TOC PDF (adjust if needed)


    # Step 1: Create smaller PDF containing the first few pages (TOC)
    pdfCreationResult = createSmallerPdf(pdfPath, tocPdfOutputPath, numPages=numPagesForTOC)
    print(pdfCreationResult) # Print result of PDF creation

    if pdfCreationResult.startswith("Error"):
        print("Stopping script due to smaller PDF creation error.") # Stop if PDF creation failed
    else:
        # Step 2: Use Gemini API (google.genai) to extract chapter info directly from the smaller PDF
        gemini_chapter_info = extractChapterInfoGeminiPdfDirect(tocPdfOutputPath)

        if gemini_chapter_info.startswith("Error"):
            print(gemini_chapter_info) # Print Gemini API error message
        else:
            # Step 3: Save Gemini's chapter information to a file
            try:
                with open(chapterInfoOutputPath, 'w', encoding='utf-8') as chapterInfoFile:
                    chapterInfoFile.write(gemini_chapter_info)
                print(f"Chapter information from Gemini (from PDF Direct API) saved to: {chapterInfoOutputPath}")
                print("\n--- Extracted Chapter Information from Gemini (from PDF Direct API) ---\n")
                print(gemini_chapter_info) # Print Gemini output to console
            except Exception as e:
                print(f"Error saving chapter information to file: {e}")