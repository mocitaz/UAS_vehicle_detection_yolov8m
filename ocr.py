from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Path to tesseract.exe

image = Image.open('image.jpg') # Replace 'image.jpg' with the actual image file path

text = pytesseract.image_to_string(image)

print(text)