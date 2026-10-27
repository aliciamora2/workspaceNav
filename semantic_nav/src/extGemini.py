import sys
import os
import google.generativeai as genai
import PIL
import json
import re

def main():
    os.environ['GOOGLE_API_KEY'] = "KEY_HERE"
    genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
    model = genai.GenerativeModel("gemini-2.5-flash")

    frase = sys.argv[1]

    response = model.generate_content([frase])
    print(response.text.strip())
    #sentences = re.findall(r'"(.*?)"', response_user.text)
    #print(sentences)

if __name__ == "__main__":
    main()
