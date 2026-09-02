def generate_temporary_answer(course: str, question: str,style: str, include_example: bool,) -> str:
    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    answer = (
        f"This is a temporary {style.lower()} explanation "
        f"for the course {course}.\n\n"
        f"You asked: {clean_question}"
    )

    if include_example:
        answer += "\n\nExample: A relevant example will appear here."

    return answer


