import os
import sys
import google.generativeai as genai
import PIL.Image

def main():
    user_input = sys.argv[1]
    image_path = sys.argv[2]

    os.environ['GOOGLE_API_KEY'] = "KEY_HERE"
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

    try:
        img = PIL.Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    model = genai.GenerativeModel("gemini-2.5-flash")

    # Generar respuesta
    response = model.generate_content([user_input, img])
    print(response.text.strip())

if __name__ == "__main__":
    main()
