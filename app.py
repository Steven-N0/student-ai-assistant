import streamlit as st

from pdf_service import (
    extract_pdf_text,
    chunk_pages,
    find_relevant_chunks,
    build_context,
    add_embeddings_to_chunks,
)

from ai_service import generate_ai_answer


st.set_page_config(
    page_title="Student AI Assistant",
    page_icon="🎓",
    layout="centered",
)



api_key = st.secrets["OPENAI_API_KEY"]
model_name = st.secrets["OPENAI_MODEL"]




if "messages" not in st.session_state:
    st.session_state.messages = []

if "document" not in st.session_state:
    st.session_state.document = None

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []



st.title("Student AI Assistant")

st.write(
    "Ask questions, generate study materials, "
    "and receive explanations adapted to your needs."
)

st.divider()




courses = [
    "Operating Systems",
    "Data Structures",
    "Differential Equations",
    "General Question",
]




st.subheader("Course document")

uploaded_pdf = st.file_uploader(
    "Upload one PDF",
    type=["pdf"],
    help="Upload lecture notes, slides, or another course PDF.",
)


if uploaded_pdf is not None:

    current_document_name = (
        st.session_state.document["name"]
        if st.session_state.document
        else None
    )

    if uploaded_pdf.name != current_document_name:

        try:
            with st.spinner("Reading and processing PDF..."):

                pdf_data = extract_pdf_text(uploaded_pdf)

                chunks = chunk_pages(
                    pdf_data["pages"],
                    1000,
                    200,
                )

                embedded_chunks = add_embeddings_to_chunks(
                    chunks,
                    api_key,
                )

                st.session_state.document_chunks = embedded_chunks

                st.session_state.document = {
                    "name": uploaded_pdf.name,
                    "full_text": pdf_data["full_text"],
                    "pages": pdf_data["pages"],
                    "page_count": pdf_data["page_count"],
                }

            st.success("PDF processed successfully.")

        except Exception as error:

            st.session_state.document = None
            st.session_state.document_chunks = []

            st.error(f"Could not process the PDF: {error}")




if st.session_state.document:

    document = st.session_state.document

    st.write(f"**File:** {document['name']}")
    st.write(f"**Pages:** {document['page_count']}")

    st.write(
        f"**Extracted characters:** "
        f"{len(document['full_text']):,}"
    )

    with st.expander("Preview extracted text"):

        preview_length = 3000

        preview = document["full_text"][:preview_length]

        st.text(preview)

        if len(document["full_text"]) > preview_length:
            st.caption(
                "Preview limited to the first 3,000 characters."
            )

    if st.button("Remove document"):

        st.session_state.document = None
        st.session_state.document_chunks = []

        st.rerun()




with st.form("question_form"):

    selected_course = st.selectbox(
        "Select a course",
        courses,
    )

    explanation_style = st.radio(
        "Choose an explanation style",
        [
            "Simple",
            "Detailed",
            "Step-by-step",
            "Exam revision",
        ],
    )

    include_example = st.checkbox(
        "Include an example"
    )

    question = st.text_area(
        "Enter your question",
        placeholder="Example: What is process synchronization?",
        height=150,
        max_chars=2000,
    )

    submit_button = st.form_submit_button(
        "Ask Assistant",
        use_container_width=True,
    )




st.divider()
st.subheader("Conversation")


if not st.session_state.messages:

    st.info(
        "No messages yet. Submit a question to begin."
    )

else:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])




if st.session_state.messages:

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()




if submit_button:

    try:

        clean_question = question.strip()

        if not clean_question:
            raise ValueError(
                "Question cannot be empty."
            )

        with st.spinner("Generating an answer..."):

            context = ""

            if st.session_state.document_chunks:

                best_chunks = find_relevant_chunks(
                    clean_question,
                    st.session_state.document_chunks,
                    api_key,
                    top_k=3,
                )

                context = build_context(
                    best_chunks
                )

            answer = generate_ai_answer(
                api_key=api_key,
                model=model_name,
                course=selected_course,
                question=clean_question,
                style=explanation_style,
                include_example=include_example,
                conversation_history=st.session_state.messages,
                context=context,
            )


        st.session_state.messages.append(
            {
                "role": "user",
                "content": clean_question,
            }
        )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


        st.rerun()


    except ValueError as error:

        st.warning(str(error))


    except Exception as error:

        st.error(
            "The assistant could not generate an answer."
        )

        print(
            f"AI request error: {error}"
        )