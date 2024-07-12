from llama_cpp import Llama

# Load the GGUF model
model_path = "llama-7b.Q4_K_M.gguf"  # Replace with your model path
llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1)  # -1 uses all available GPU layers

def generate_answer(query, contexts):
    context_text = "\n".join([f"Source {i+1}: {content[:500]}..." for i, (_, content, _) in enumerate(contexts)])
    
    prompt = f"""Context information is below.
---------------------
{context_text}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer: """
    
    response = llm(prompt, max_tokens=200, stop=["Query:", "\n\n"], echo=False)
    return response['choices'][0]['text'].strip()
