import fitz

def is_scanned_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    
    for page_number in range(doc.page_count):
        page = doc[page_number]
        text = page.get_text("text")
        
        # Check if the page has text content
        if text.strip():
            print(f"Page {page_number + 1} contains text. This PDF is likely not scanned.")
            return False

    print("No text found in the PDF. This PDF is likely scanned.")
    return True

# Example usage:
pdf_path = "C:/Users/hp/Desktop/PDF/New folder/priciple willingness letter_compressed.pdf"
is_scanned = is_scanned_pdf(pdf_path)

if is_scanned:
    print("The PDF is likely scanned.")
else:
    print("The PDF is likely not scanned.")