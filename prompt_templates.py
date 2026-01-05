from langchain_core.prompts import PromptTemplate

prev_template = """
look at the below passage and see if it is relevant to this: {question}.
passage: {passage}

Dont make things up.
condense it with the relevant information only. And do not make up anything if no relevant information found, just say -.
"""

final_template = """
look at all the below passges and answer the question: {question}
Dont make things up.

Passages :-
{all_passages}
"""

prev_prompt = PromptTemplate.from_template(prev_template)
final_prompt = PromptTemplate.from_template(final_template)

def get_prev_and_final_prompt():
    return prev_prompt, final_prompt