import google.generativeai as genai

genai.configure(api_key="AIzaSyCy_dNG2bqMl9Xdj57yVXRVzNRLuLeAB6s")

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content("Explain AI in simple terms")

print(response.text)