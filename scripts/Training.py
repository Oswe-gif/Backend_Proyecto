import os
import google.generativeai as genai
import PIL.Image
import glob


genai.configure(api_key="AIzaSyDMywIL8s5o_OpHfUD4wbL6cwndpD4azsY")

model = genai.GenerativeModel("gemini-1.5-flash")

product_image_paths = glob.glob("Smallest/*.jpg")

#product_image_paths = product_image_paths[:5]

def upload_product_images(image_paths):
    product_responses = {}
    for image_path in image_paths:
        try:
            product_image = PIL.Image.open(image_path)
            response = model.generate_content(["Describe this product, name and general information and tell the advantages and disadvantages, but pls be short", product_image])
            product_responses[image_path]=response.text
        except Exception as e:
            print(f"Error al procesar la imagen {image_path}: {e}")
    return product_responses

print(upload_product_images(product_image_paths))