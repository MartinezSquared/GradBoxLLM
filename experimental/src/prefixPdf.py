
import PyPDF2

def prefixPdf(pdfPath, outputPath, numPages):
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

if __name__ == "__main__":
    pdfInput = input("Enter input path to original PDF file: ")
    pdfOutput = input("Enter output path for the smaller PDF file: ")
    numPages = 20
    prefixPdf(pdfInput, pdfOutput, numPages)