from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
import google.generativeai as genai
import json
sys.path.append('.')

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args
from io import BytesIO

genai.configure(api_key="AIzaSyDMywIL8s5o_OpHfUD4wbL6cwndpD4azsY")

class ImageProcessorApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.model = model = genai.GenerativeModel("gemini-1.5-flash")
        self.diccionario_datos = {'Smallest\\mate.jpg': 'This is a matte lipstick from the brand "Studio Look" by "cyzone". It comes in a black tube with a white label. \n\n**Advantages:**\n- Long-lasting\n- Matte finish\n\n**Disadvantages:**\n- Can be drying\n- Can be difficult to remove', 'Smallest\\name.jpg': 'This is a set of two skincare products from the brand Cyzone called Skin First Hello Age Prevent. The first product is a  facial cleanser in gel, which promises immediate cleansing, maintains pH balance, and is suitable for all skin types. The second product is a face cream that claims to provide 24 hours of hydration, brighten skin, and help prevent signs of aging.\n\n**Advantages:**\n\n*  Both products are dermatologically tested and claim to be suitable for all skin types.\n*  The gel cleanser promises immediate cleaning and balanced pH. \n*  The face cream claims to provide long-lasting hydration and help prevent signs of aging.\n\n**Disadvantages:**\n\n*  The effectiveness of these products is subjective and may vary depending on individual skin types and concerns.\n*  There is no information on the specific ingredients used, which may be a concern for some consumers.\n*  The products are advertised in Spanish, which might limit their accessibility to non-Spanish speakers.', 'Smallest\\Skinfirst_2024.jpg': 'This is a face cleanser called "Skin First Hello Age Prevent" by Cyzone. It\'s a gel cleanser that claims to balance your skin\'s pH. It\'s suitable for all skin types. \n\n**Advantages:**\n* Dermatologically tested\n* For all skin types\n\n**Disadvantages:**\n*  May not be suitable for sensitive skin.\n* No information on specific ingredients.', 'Smallest\\sl4.jpg': 'This is a set of matte liquid lipsticks from the brand "Studio Look". The lipsticks come in a variety of colors and are designed to provide long-lasting wear.\n\n**Advantages:**\n\n* Matte finish\n* Long-lasting wear\n* Variety of colors\n\n**Disadvantages:**\n\n* Can be drying\n* Can be difficult to remove\n* Not as comfortable as other lipsticks', 'Smallest\\studiolook_2024.jpg': "The Studiolook Matte lipstick comes in a sleek black tube with a pink liquid inside. It offers a long-lasting matte finish with a wide range of colors. It's known for its high pigment and intense color payoff. \n\n**Advantages:**\n*  Long-lasting wear\n*  Matte finish\n*  Intense color \n\n**Disadvantages:**\n*  Can be drying on the lips\n*  May emphasize dry patches \n*  Difficult to remove completely\n", 'Smallest\\sweetblack_2024.jpg': 'This is a perfume bottle called "Sweet Black". It is a dark, mysterious fragrance with a pink-tinted glass bottle and a black cap. It is meant to be a seductive and alluring scent.\n**Advantages:** \n* It is a long-lasting and complex fragrance.\n* The bottle is elegant and unique.\n**Disadvantages:**\n* The scent may be too strong for some people.\n* The price may be high.', 'Smallest\\YCP2313TD312P04H1.jpg': "This is the Cyzone Skin First line of products, focused on anti-aging.\n\n**Skin First Limpiadora facial en gel:**\n- A face cleanser gel that provides immediate cleaning and keeps the pH balanced.\n- It's suitable for all skin types.\n**Advantages:**\n- Gentle and effective\n- Balances skin pH\n**Disadvantages:**\n-  None mentioned\n\n**Skin First Hidratante facial en crema:**\n- A face cream that provides 24 hours of hydration and helps prevent the first signs of aging.\n- Suitable for all skin types.\n**Advantages:**\n- Long-lasting hydration\n- Helps prevent aging\n**Disadvantages:**\n- None mentioned", 'Smallest\\YMQ2301CL30PA1L0.jpg': 'This is a Cyzone Cyplay lip crayon, a type of makeup designed for lips. It offers a creamy texture and a pigmented color payoff. \n\n**Advantages:**\n\n* Easy to apply and blend\n* Long-lasting\n* Provides a smooth finish\n* Various shades available\n\n**Disadvantages:**\n\n* Can be drying on lips if not properly moisturized beforehand\n* Might be difficult to sharpen\n* Can smudge easily if not set with a lip liner or powder\n* Not suitable for all skin types\n', 'Smallest\\YMQ2304TD42P02L3.jpg': 'The product is a nail polish from the brand Cyzone, called "CyPlay". It comes in various colours. \n\n**Advantages:**\n*  It\'s affordable and comes in many colours.\n\n**Disadvantages:** \n*  The quality may not be as high as other brands. \n*  The colours might not be as vibrant or long-lasting as other nail polishes. \n', 'Smallest\\YMQ2308TD62P01H0.jpg': "The product is a lip balm called **Cyzone** Cyplay Lip Balm.  \n\n**General Information:**\n\n- It's a lip balm that comes in a variety of colors. \n- It is available in 4 shades: **Pink, Coral, Nude and Red**.\n- It is intended to provide hydration and color to the lips.\n\n**Advantages:**\n\n- It provides long-lasting color and hydration. \n- It is very moisturizing.\n- It's affordable.\n\n**Disadvantages:**\n\n- The color may not be as vibrant as lipstick.\n- It can be drying if it's not reapplied frequently. \n", 'Smallest\\YMQ2309TD56P01L0.jpg': 'The product is a set of three eye pencils from the brand "GYPlay". They are called "CYZONE Eye Lines" and come in the colors grey, blue and white. They are said to be smudge-proof and long-lasting.\nAdvantages: Easy to apply, smudge-proof, long-lasting.\nDisadvantages: Not available in other colors.', 'Smallest\\YMQ2313TD50P01B3.jpg': 'The product is a lip gloss by the brand **CYZONE**. It comes in a variety of colors, including red, pink, purple, and white. The gloss is designed to give lips a glossy finish with a hint of shimmer.\n\n**Advantages:**\n\n* Provides a high-shine finish.\n* Comes in a variety of colors.\n*  Can be applied over lipstick for extra shine.\n\n**Disadvantages:**\n\n*  May be sticky.\n* May smudge easily.\n*  Not long-lasting. \n', 'Smallest\\YTF2311TD128P01C0.jpg': 'The product is called "Skin First Purificante Exfoliante Facial en Crema" by Cyzone. It\'s a facial exfoliating cream that claims to unclog pores and improve the appearance of pimples and blemishes. The advantage of this product is that it helps with skin clarity and texture. However, it may be too strong for sensitive skin and can cause irritation. \n', 'Smallest\\YTF2316TD210P01L01.jpg': "This is the Cyzone Skin First Eye Detox Effect, a cream gel eye contour in a roll-on format. It's designed to instantly reduce dark circles and puffiness under the eyes. \n\n**Advantages:**\n* Roll-on applicator for easy application\n* Contains caffeine and vitamin B3 to help reduce dark circles\n* Cooling effect helps to reduce puffiness\n\n**Disadvantages:**\n* May not be suitable for sensitive skin\n* Not a long-term solution for dark circles\n* Can be expensive compared to other eye creams", 'Smallest\\200039855_fotoproducto.jpg': 'The product is a men\'s fragrance called "Nitro" by cy\'zone.  The bottle is black with a silver cap. The fragrance is described as having a strong and masculine scent.\n\n**Advantages:**\n\n* Strong and masculine scent.\n* Attractive packaging.\n\n**Disadvantages:**\n\n* May be too strong for some people.\n* Not very long-lasting.', 'Smallest\\200060760_fotoproducto.jpg': "This is Nitro Air cologne by cy'zone. It's a masculine scent designed to be fresh and invigorating.  It's a popular choice for everyday wear, but it's not a strong fragrance.  It can be long-lasting but that may vary based on skin type. The bottle is aesthetically pleasing and the price is affordable. \n", 'Smallest\\200086172_fotoproducto.jpg': "This is Nitro Intense by Cyzone, a men's fragrance. It's a bold, woody scent with notes of bergamot, cardamom, cedarwood, and amber.\n\n**Advantages:**\n\n* Long-lasting scent\n* Masculine and sophisticated\n* Affordable price\n\n**Disadvantages:**\n\n* Can be overpowering for some\n* Not as versatile as other fragrances\n*  Not available in all regions.", 'Smallest\\200092333_fotoextra.jpg': 'This is Ainnara perfume, a product by Cyzone. It is known for its floral and fruity scent, with notes of bergamot, mandarin orange, rose, and jasmine.\n\n**Advantages:**\n\n* Pleasing and feminine scent.\n* Long-lasting fragrance.\n* Affordable price.\n\n**Disadvantages:**\n\n* Some may find the scent too sweet or overpowering.\n* May not be suitable for all skin types.', 'Smallest\\200092333_fotoproducto.jpg': 'The product is a perfume called "Ainnara" by Cyzone. It\'s a delicate floral scent with a sophisticated and elegant vibe. It\'s designed for women and comes in a beautiful bottle. \n\nAdvantages: \n*  It\'s a long-lasting and pleasant fragrance. \n*  The packaging is elegant and visually appealing. \n*  The fragrance is known for its delicate floral scent. \n\nDisadvantages: \n*  The perfume can be expensive compared to other brands. \n*  It might not be suitable for everyone, especially those who prefer strong and bold scents. \n*  The fragrance is not as well-known as other popular perfume brands. \n', 'Smallest\\200092742_fotoproducto.jpg': "This is Blue & Blue for Him, a men's fragrance by Cyzone. It comes in a blue glass bottle with a silver cap and sprayer. \n\n**Advantages:**  \n- Fresh and masculine scent.\n- Affordable price.\n\n**Disadvantages:**\n- May not be long-lasting.\n- Not a unique scent.", 'Smallest\\200092744_fotoproducto.jpg': 'This is a bottle of Blue & Blue+ perfume for women. \n\n**Advantages**: \n*  It is a fresh and clean smelling fragrance. \n* It is affordable. \n\n**Disadvantages**: \n*  It is not a long-lasting fragrance. \n* It is not a very complex or unique scent.', 'Smallest\\200094936_fotoproducto.jpg': "This is a bottle of Cyzone Go Taste Cool Cologne. It is a refreshing cologne with a sweet, fruity scent. \n\nAdvantages:\n\n* It has a pleasant smell.\n* It's refreshing.\n* It comes in a stylish bottle.\n\nDisadvantages:\n\n* It may not last as long as other colognes.\n* The scent may be too sweet for some people. \n", 'Smallest\\200094941_fotoproducto.jpg': "This is a bottle of Cyzone's Taste Warm cologne. It is a refreshing cologne that is perfect for everyday wear. It has a sweet and warm scent that is not overpowering.\n\nAdvantages:\n- Refreshing and invigorating.\n- Sweet and warm scent.\n- Not overpowering.\n\nDisadvantages:\n- May not be suitable for everyone's taste.\n-  It is not a long-lasting fragrance.", 'Smallest\\200095911_fotoproducto.jpg': 'This is a watermelon-scented body spray called "Sandia Shake" by Cyzone. It\'s a refreshing colonia that comes in a 240 ml bottle. \n\n**Advantages:**\n- Smells like watermelon\n- Refreshing scent\n- Good for summer\n\n**Disadvantages:**\n- May not be strong enough for some people\n- Not a long-lasting scent\n-  Not everyone enjoys watermelon scents', 'Smallest\\200095964_fotoproducto.jpg': 'This is a  "Pera in Love" cologne by Cyzone. It\'s a refreshing cologne with a pear scent.\n\n**Advantages:**\n\n* Pleasant, fruity scent\n*  Refreshing \n*  Affordable\n\n**Disadvantages:**\n\n* May not last long\n* May be too sweet for some \n', 'Smallest\\200097947_fotoproducto.jpg': 'This is a bottle of Sweet Black Intense perfume. It is a dark, sensual, and captivating fragrance that is perfect for evening wear. \n\n**Advantages:**\n- Long-lasting scent\n- Unique and attractive bottle design\n- Good value for the price\n\n**Disadvantages:**\n- Can be overpowering if overused\n- May not be suitable for all tastes', 'Smallest\\200099686_fotoproducto.jpg': 'This is a body mist called "Taste Pop" by the brand "Victoria\'s Secret". It has a sweet candy scent and is designed to be a refreshing body spray. \n\n**Advantages:**\n* Sweet and refreshing scent\n* Convenient spray bottle\n* Affordable price\n\n**Disadvantages:**\n* Scent may not be strong enough for some people\n* May not last very long on the skin\n* Can be drying for some skin types', 'Smallest\\lipbalm.jpg': 'This is a Cyzone "Cy Play" lip gloss. It is a tinted lip gloss that gives a glossy finish. It is available in different shades and has a light, comfortable formula. \n\n**Advantages:**\n*  Provides a glossy finish\n*  Available in different shades\n*  Light and comfortable formula\n\n**Disadvantages:**\n* May not be as long-lasting as other lip products\n*  May transfer to other surfaces \n', 'Smallest\\Pure.jpg': 'The product is a fragrance from the brand Cyzone. It comes in two versions: Pure Bloom and Pure Vibes.\n\n**Advantages:**\n\n* Attractive packaging.\n* Pleasant scents. \n\n**Disadvantages:**\n\n* Limited information about the scent notes.\n* No information about the longevity or sillage of the fragrances.'}

    def setup_routes(self):
        @self.app.route('/process_image', methods=['POST'])
        def process_image():
            return self.handle_process_image()

    def decode_image(self, base64_str):
        """Decode a base64 image string to a NumPy array (OpenCV image)."""
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data))
        return img

    def encode_image(self, img):
        """Encode an OpenCV image to a base64 string."""
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_image
    
    def reccomend_products(self, img):
        recommendations = ""
        diccionario = "Lista de productos:\n"
        for key, value in self.diccionario_datos.items():
            diccionario += f"{key}: {value}\n"
        message = ["Mira el siguiente diccionario, quiero que selecciones los 8 mejores productos entre todos de acuerdo al rostro enviado como imagen y me devueltas en un formato JSON, la key del producto, nombre del producto, inventa precio, la razon de seleccion, pls sé breve. Esta es la cara, recuerda dar recomendaciones si es hombre o mujer, estado de piel, color de piel, etc: ", img, " eston son los productos ", diccionario, "Por favor solo el json, ya sé que se debe ir al medico"]

        recommendations = self.model.generate_content(message)
        return recommendations.text[7:-3]
    
    def get_image_from_color(self, color):
        source_dir = "assets/images/makeup"
        if color=="white":
            return Image.open(os.path.join(source_dir, "makeup_2.jpg")).convert('RGB')
        if color=="purple":
            return Image.open(os.path.join(source_dir, "makeup_1.jpg")).convert('RGB')
        if color=="pink":
            return Image.open(os.path.join(source_dir, "makeup_3.jpg")).convert('RGB')
        print("The color was not found")
    
    def expand_canvas(self, img, new_width, new_height):
        """Expand the canvas of a PIL image with white color."""
        # Create a new image with the desired size and white background
        new_image = Image.new('RGB', (new_width, new_height), color='white')
        
        # Calculate the position to paste the original image
        paste_x = (new_width - img.width) // 2
        paste_y = (new_height - img.height) // 2

        # Paste the original image onto the new canvas
        new_image.paste(img, (paste_x, paste_y))

        return new_image


    def handle_process_image(self):
        data = request.json

        # Decode the base64 image
        try:
            img_person = self.decode_image(data['img'])
            color = data['color']
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        
        
        jsonData = json.loads(self.reccomend_products(img_person))

        for i in range(len(jsonData)):
            try:
                with open(jsonData[i]["key"], 'rb') as imagen_file:
                    imagen_base64 = base64.b64encode(imagen_file.read()).decode('utf-8')
                    jsonData[i]["key"] = imagen_base64
            except:
                pass

        original_width, original_height = img_person.size
        color_img = self.get_image_from_color(color)
        if original_height<original_width:
            new_height = min(800, original_height)
            aspect_ratio = original_height / new_height
            new_width = int(original_width/aspect_ratio)
            img_person.resize((new_width, new_height))
            if new_height<800:
                color_img = color_img.resize((new_height, new_height))
            color_img = self.expand_canvas(color_img, new_width, new_height)
        else:
            new_width = min(800, original_width)
            aspect_ratio = original_width / new_width
            new_height = int(original_height/aspect_ratio)
            img_person.resize((new_width, new_height))
            if new_width<800:
                color_img = color_img.resize((new_width, new_width))
            color_img = self.expand_canvas(color_img, new_width, new_height)

        # Perform OpenCV processing on `img`
        result_makeup = self.inference.transfer(img_person, color_img, postprocess=True)

        # Encode the processed image back to base64
        print(result_makeup)
        base64_image = self.encode_image(result_makeup)

        return jsonify({'makeup_image': base64_image, 'recommendations': jsonData})

    def run(self, config, args):
        logger = create_logger(args.save_folder, args.name, 'info', console=True)
        print_args(args, logger)
        logger.info(config)

        self.inference = Inference(config, args, args.load_path)
        self.app.run(ssl_context=("cert.pem", "key.pem"), debug=True, host="0.0.0.0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model", 
                        default='ckpts/sow_pyramid_a5_e3d2_remapped.pth')

    parser.add_argument("--source-dir", type=str, default="assets/images/non-makeup")
    parser.add_argument("--reference-dir", type=str, default="assets/images/makeup")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = "cpu"
    args.device = torch.device("cpu")

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    config = get_config()
    app_instance = ImageProcessorApp()
    app_instance.run(config, args)