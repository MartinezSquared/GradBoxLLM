import PyPDF2

def createSmallerPdf(pdfPath, outputPath, numPages=20):
    """
    Creates a new PDF containing the first 'numPages' pages of the input PDF.

    Args:
        pdfPath (str): Path to the input PDF file.
        outputPath (str): Path to save the new smaller PDF file.
        numPages (int): Number of pages to include in the new PDF (default 10).

    Returns:
        str: A success message indicating the output file path,
             or an error message if the input PDF is not found or an error occurs during PDF creation.
    """
    try:
        with open(pdfPath, 'rb') as pdfFile:
            pdfReader = PyPDF2.PdfReader(pdfFile)
            pdfWriter = PyPDF2.PdfWriter()
            numPagesToCopy = min(numPages, len(pdfReader.pages))

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


if __name__ == "__main__":
    pdfPath = "../googleDrive/assets/textbooks/medicalSurgicalNursing-ConceptsAndPractice.pdf"  # Replace with the actual path to your PDF
    outputPath = "../googleDrive/assets/chuncks/medicalSurgicalNursing-ConceptsAndPractice_10pages.pdf" # Replace with your desired output file path
    numPagesForSmallerPdf = 10 # You can change this to create a PDF with a different number of pages

    resultMessage = createSmallerPdf(pdfPath, outputPath, numPages=numPagesForSmallerPdf)
    print(resultMessage)