from openai import OpenAI


def generate_ai_answer(
    api_key: str,
    model: str,
    course: str,
    question: str,
    style: str,
    include_example: bool,
    conversation_history: list[dict[str, str]],
    context: str,
) -> str:

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    if not api_key.strip():
        raise ValueError("The OpenAI API key is missing.")

    if not model.strip():
        raise ValueError("The OpenAI model name is missing.")


    # --------------------------------------------------
    # OPENAI CLIENT
    # --------------------------------------------------

    client = OpenAI(api_key=api_key)




    if include_example:
        example_instruction = (
            "Include one clear and relevant example."
        )
    else:
        example_instruction = (
            "Do not include an example unless it is necessary "
            "to answer the question correctly."
        )




    instructions = f"""
You are an AI study assistant for university students.

The student's selected course is: {course}
The requested explanation style is: {style}

Follow these rules:
- Answer the student's question clearly and accurately.
- Adapt the depth and wording to the selected explanation style.
- Use headings or steps when they improve clarity.
- Use the provided PDF context when it is relevant to the question.
- If the student's question is clearly unrelated to the selected course,
  briefly tell them which course it seems to belong to and ask them to switch
  to the appropriate course before answering.
- If PDF context is provided, prioritize that information when answering questions about the uploaded document.
- If the provided PDF context does not contain enough information, clearly say so.
- Do not invent information and claim that it came from the PDF.
- {example_instruction}
"""




    recent_history = conversation_history[-10:]

    input_messages = recent_history.copy()




    if context:
        input_messages.append(
            {
                "role": "user",
                "content": (
                    "The following text was retrieved from the "
                    "student's uploaded PDF. Use it when it is "
                    "relevant to the question.\n\n"
                    f"{context}"
                ),
            }
        )




    input_messages.append(
        {
            "role": "user",
            "content": clean_question,
        }
    )




    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_messages,
    )




    answer = response.output_text.strip()

    if not answer:
        raise RuntimeError(
            "The AI returned an empty response."
        )

    return answer