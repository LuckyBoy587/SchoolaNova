import numpy as np
import faiss
import re
import traceback
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class SimpleRag:
    """
    A simple Retrieval-Augmented Generation (RAG) system.

    This class handles the following steps:
    1. Reads a text file and cleans the content.
    2. Creates and stores sentence embeddings using SentenceTransformer and FAISS.
    3. Retrieves the most relevant text snippets for a given query.
    4. Generates questions based on the retrieved text using a T5 model.
    """

    def __init__(self, fileName):
        self.fileName = fileName
        self.lines = []
        # Models and index are lazily loaded to improve startup time
        self.model = None
        self.index = None
        self.question_model = None
        self.rephrase_model = None

        # Call setup methods to prepare the data and embeddings
        self._intialize()

        def _initialize(self):
            """Initializes the RAG system by loading data, embeddings, and models."""
            self.getText()
            self.createEmbeddings()
            self._load_generation_models()

    def getText(self):
        """Reads the text file and preprocesses the content."""
        try:
            with open(self.fileName, 'r', encoding='utf-8') as file:
                self.book_data = file.read()
                
            temp_lines = self.book_data.split("\n")

            # Remove numbers and clean up lines
            temp_lines = [re.sub(r' +', ' ', line.strip()) for line in temp_lines]
            self.lines = [line for line in temp_lines if line]

            print(f"Total Lines extracted: {len(self.lines)}")
        except FileNotFoundError:
            print(f"Error: The file '{self.fileName}' was not found.")
            traceback.print_exc()
        except Exception as e:
            print(f"Error reading file: {e}")
            traceback.print_exc()

    def createEmbeddings(self):
        """Creates embeddings for the text lines and builds a FAISS index."""
        try:
            # Initialize Sentence-Transformer model if not already loaded
            if self.model is None:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Initialized embedding model.")

            # Create embeddings from book_data
            embeddings = self.model.encode(self.lines, normalize_embeddings=True).astype('flaot32') # FAISS requires float32

            # Create a FAISS index
            d = embeddings.shape[1]  # Dimension of the embeddings
            self.index = faiss.IndexFlatL2(d)
            self.index.add(embeddings)

            print("Created embeddings and FAISS index.")
            print(f"Total Number of vectors in index: {self.index.ntotal}")
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            traceback.print_exc()

    def _load_generation_models(self):
        """Helper to load all heavy generation models."""
        try:
            self.question_model = pipeline(
                "text2text-generation",
                model="mrm8488/t5-base-finetuned-question-generation-ap"
            )
            print("Question generation model loaded.")
        except Exception as e:
            print(f"Error loading question generation model: {e}")
            traceback.print_exc()

        try:
            self.rephrase_model = pipeline(
                "summarization",
                model="t5-small"
            )
            print("Rephrasing model loaded.")
        except Exception as e:
            print(f"Error loading rephrasing model: {e}")
            traceback.print_exc()


    def get_relevent_lines(self, q, k=3):
        """
        Retrieves the top k most relevant text snippets for a given query.

        Args:
            q (str): The search query.
            k (int): The number of top results to retrieve.

        Returns:
            list: A list of dictionaries, each containing 'rank', 'score', and 'text'.
        """
        if self.model is None or self.index is None:
            print("Model or index not initialized. Cannot get answer.")
            return []

        try:
            # Create embeddings for the query
            query_embedding = self.model.encode([q], normalize_embeddings=True).astype('flaot32')

            # Search the FAISS index for relevant lines
            distances, indices = self.index.search(query_embedding, k)

            result = []
            for i in range(k):
                # Ensure the index is valid before accessing
                if indices[0][i] < len(self.lines):
                    result.append(self.lines[indices[0][i]])

            return result
        except Exception as e:
            print(f"Error getting answer: {e}")
            traceback.print_exc()
            return []

    def get_questions(self, chapter, n):
        """
        Generates questions based on the most relevant text related to a chapter.

        Args:
            chapter (str): The chapter topic to search for.
            n (int): The number of questions to generate.

        Returns:
            list: A list of generated question strings.
        """
        if self.question_model is None:
            print("Question generation model not loaded. Cannot generate questions.")
            return []

        try:
            # Get relevant lines from the book
            relevant_snippets = self.get_relevent_lines(chapter, 5)

            if not relevant_snippets:
                print("No relevant text found to generate questions.")
                return []
            
            # Prompt the query for question generation  [Model expects -> 'generate question: {answers}']
            answers_text = " ".join([item for item in relevant_snippets])
            formatted_input = f"generate question: {answers_text}"

            print(f"Number of snippets used for generation: {len(relevant_snippets)}")

            questions = self.question_model(formatted_input, max_length=64, num_return_sequences=n, do_sample=True, top_k=50)

            # return list of generated questions
            return [q["generated_text"] for q in questions]

        except Exception as e:
            print(f"Error getting questions: {e}")
            traceback.print_exc()
            return []

    def rephrse_answer(self, answers):
      """
        Rephrases the given text using the summarization model.

        Args:
            answers (str): The text to be rephrased.

        Returns:
            str: The rephrased text.
        """
      print(f"Original Text: \n {answers}\n\n\n")
      if self.rephrase_model is None:
        print("Rephrasing model not loaded. Cannot rephrase answer.")
        return ""

      try:
        # Initializing the rephrase model
        summary = self.rephrase_model(
            answers,
            max_length=len(answers.split()) + 10,
            min_length=len(answers.split()) - 10,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9
        )

        # return the summarized text
        return summary[0]['summary_text']
      except Exception as e:
        print(f"Error rephrasing answer: {e}")
        traceback.print_exc()

    # helper function to use the inner_functions
    def get_answer(self, query, k=3):
      raw_text = "".join(self.get_relevent_lines(query, k))
      return self.rephrse_answer(raw_text)

