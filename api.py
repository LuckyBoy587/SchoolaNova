# pip install google.generativeai pillow

import google.generativeai as genai

API_KEY = ""

question = _getQuestion()
lines = _getLines()


genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')

prompt = f"""
You are a highly efficient text-analysis AI. Your task is to answer the question below using ONLY the information found in the "RELEVANT LINES" section.

**Instructions & Constraints:**
1.  Read the question carefully.
2.  Analyze the "RELEVANT LINES" to find the answer.
3.  Your answer must be based exclusively on the provided text. Do not use any external knowledge.
4.  If the answer cannot be found in the lines, state exactly that: "The answer is not available in the provided lines."
5.  Provide only the direct answer without any extra conversation.

---

**## QUESTION:**
{question}

---

**## RELEVANT LINES:**
{lines}"""

try:
    response = model.generate_content(prompt)

    print("Result\n\n\n")
    print(response.text)

except Exception as e:
    print(f"An error occured: {e}")
