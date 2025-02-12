import os

def pdfPathDict(path):
    """
    Read all PDF files in a folder and return a dictionary of PDF file paths.
    Args:
        path (str): Path to the folder containing PDF files.
    Returns:
        dict: Key = filename (without extension)
              Value =  The full path to the PDF file.
    """
    pdf_paths = {}
    try:
        for item in os.listdir(path):
            if os.path.isfile(os.path.join(path, item)) and item.endswith('.pdf'):
                filepath = os.path.join(path, item)
                absolute_filepath = os.path.abspath(filepath) # Convert to absolute path here
                filename_no_ext = os.path.splitext(item)[0]
                pdf_paths[filename_no_ext] = absolute_filepath
    except FileNotFoundError:
        print(f"Error: Folder not found at path: {path}")
        return {}
    except Exception as e:
        print(f"Error: {e}")
        return {}

    return pdf_paths

if __name__ == "__main__":
    path = input("Enter folder path containing PDFs: ")

    pdfPathDict(path)