import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox

def extract_images_after_text(pdf_path, output_folder, specific_text):
    # Load PDF document
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page in the PDF
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]

        # Reset the flag for each new page
        text_found = False

        # Extract text on the page
        page_text = page.get_text()

        # Check if the specific text is found on the page
        if specific_text.lower() in page_text.lower():
            text_found = True

        # Extract images on the page only if the specific text has been found
        if text_found:
            images = page.get_images(full=True)

            # Iterate through each image on the page
            for img_info in images:
                image_index = img_info[0]
                image = pdf_document.extract_image(image_index)
                image_bytes = image["image"]

                # Save the image to a file
                image_filename = f"{output_folder}/page{page_number + 1}_img{image_index}.png"
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)

    # Close the PDF document
    pdf_document.close()
    
    
def is_signature(image):
    # Your existing code for signature detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    black_pixel_count = np.sum(binary_image == 0)
    threshold_black_pixels = 100000  # Example threshold
    return black_pixel_count > threshold_black_pixels



def process_images_in_folder(folder_path):
    
    root = tk.Tk()
    root.withdraw()
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):  # Add more extensions if needed
            image_path = os.path.join(folder_path, filename)
            extracted_image = cv2.imread(image_path)

            # Check if the image contains a signature
            if is_signature(extracted_image):
                messagebox.showinfo("Signature Detection", f"Signature is present")
            

# Example usage

pdf_path = "C:/Users/hp/Desktop/PDF/Amer Sports Timeclock Proposal signed.pdf"
output_folder = "C:/Users/hp/Desktop/PDF"
specific_text = "Signature:"
extract_images_after_text(pdf_path, output_folder, specific_text)


folder_path = "C:/Users/hp/Desktop/PDF"
process_images_in_folder(folder_path)