
def map_and_reduce(max_tokens: int,
                    user_query: str,
                    k: int,
                    llm,
                    vector_store,
                    prev_prompt,
                    final_prompt):
    """
    opts for 'map and reduce method' incase the entire input text exceeds max_tokens limit
    """
    
    relevant_chunks =[document.page_content 
                      for document in vector_store.similarity_search(query=user_query, k=k)]    
    
    combined_chunks = "\n".join(relevant_chunks)
    
    if len(combined_chunks) < max_tokens:
        output = llm.invoke(final_prompt.format(question=user_query, all_passages=combined_chunks))
        return output
    
    condensed_answers = [
    llm.invoke(prev_prompt.format(question=user_query, passage=chunk))
    for chunk in relevant_chunks]

    
    combined_condensed_answers = "\n".join(condensed_answers)
    output = llm.invoke(final_prompt.format(question=user_query, all_passages=combined_condensed_answers))
    return output


