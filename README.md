# Student AI Assistant

A Streamlit-based AI study assistant for university students.

The application allows students to ask course-related questions, choose an explanation style, maintain conversation history, and upload PDF course documents for context-aware answers.

## Features

- AI-powered question answering using the OpenAI API
- Course selection
- Multiple explanation styles
- Optional examples
- Conversation history
- PDF upload and text extraction
- Automatic PDF text chunking
- Semantic search using embeddings
- Retrieval of relevant PDF sections
- Context-aware AI answers based on uploaded documents
- Page number tracking for retrieved PDF content

## Technologies Used

- Python
- Streamlit
- OpenAI API
- OpenAI Embeddings
- pypdf

## How It Works

When a PDF is uploaded:

1. The PDF text is extracted page by page.
2. The text is divided into overlapping chunks.
3. An embedding is generated for each chunk.
4. The embeddings are stored during the Streamlit session.

When the student asks a question:

1. An embedding is generated for the question.
2. The question embedding is compared with the PDF chunk embeddings.
3. The most relevant chunks are selected using cosine similarity.
4. These chunks are added as context for the AI model.
5. The model generates an answer using the question, conversation history, selected course, explanation style, and relevant PDF context.

## Project Structure

```text
app.py
    Main Streamlit interface and application logic.

ai_service.py
    Handles communication with the OpenAI API and generates answers.

pdf_service.py
    Handles PDF extraction, chunking, embeddings, semantic search, and context building.