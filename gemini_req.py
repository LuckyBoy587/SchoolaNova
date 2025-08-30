from flask import Flask, request, Response, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv
import semantic_search

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

genai.configure(api_key=api_key)

app = Flask(__name__)

def generate_gemini_stream(context: str, query: str):
    """Yield streaming Gemini responses chunk by chunk."""
    contents = f"{context}\n\nQuery: {query}"
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(contents, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text

@app.route("/ask", methods=["POST"])
def ask():
    """
    POST /ask
    JSON body: { "query": "your question", "pdf_paths": ["path1.pdf", "path2.pdf"] }
    If pdf_paths not provided, uses default PDF.
    """
    data = request.get_json(force=True)
    query = data.get("query")
    pdf_paths = data.get("pdf_paths", [f"./pdfs/ch{i}.pdf" for i in range(1, 8)])

    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if not isinstance(pdf_paths, list):
        return jsonify({"error": "pdf_paths must be a list"}), 400
    
    # Collect results from all PDFs
    all_results = []
    for pdf_path in pdf_paths:
        try:
            results = semantic_search.semantic_search_pdf(
                user_query=query, 
                pdf_path=pdf_path
            )
            all_results.extend(results)
        except FileNotFoundError:
            # Skip if PDF not found
            continue
        except Exception as e:
            # Skip on other errors
            continue
    
    # Sort all results by score descending and take top 5
    all_results.sort(key=lambda x: x['score'], reverse=True)
    top_results = all_results[:5]
    
    # Extract context from top results
    if top_results:
        context_text = "\n".join([res['chunk'] for res in top_results])
    else:
        context_text = "No relevant context found in the provided PDFs."
    
    # Stream response back to client
    return Response(generate_gemini_stream(context_text, query), mimetype="text/plain")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Gemini Flask server is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
