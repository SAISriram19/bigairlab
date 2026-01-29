from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import CrossEncoder
import torch

class LLMQA:
    def __init__(self, model_name='google/flan-t5-base'):
        print(f"Loading LLM model via LangChain: {model_name}")

        device = 0 if torch.cuda.is_available() else -1
        device_name = 'GPU' if device == 0 else 'CPU'

        try:
            # Load Cross Encoder for Reranking
            print("Loading CrossEncoder for Reranking...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512, # Increased for CoT
                device=device,
                do_sample=True,
                temperature=0.3, # Lower temperature for factual accuracy
                repetition_penalty=1.15 # Prevent repetition loops
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

            # Chain of Thought Prompt
            self.prompt_template = """Answer the question based ONLY on the following context.
Follow these formatting rules:
1. Use bullet points for lists.
2. For contact info, provide the ACTUAL details (e.g., the specific phone number or address found).
3. Do not summarize what is available; list the values.

If the answer is not in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

            print(f"LangChain LLM loaded on {device_name}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def rephrase_question(self, query, chat_history):
        if not chat_history:
            return query

        history_text = ""
        for turn in chat_history[-3:]: # Look at last 3 turns
            role = "Human" if turn['role'] == 'user' else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

        prompt = f"""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{history_text}
Follow Up Input: {query}
Standalone Question:"""

        try:
            result = self.llm.invoke(prompt)
            standalone_question = result.strip()
            print(f"Original: {query} -> Rephrased: {standalone_question}")
            return standalone_question
        except Exception as e:
            print(f"Error rephrasing question: {e}")
            return query

    def generate_answer(self, query, context_chunks):
        context_text = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['content']}"
            for chunk in context_chunks[:3]
        ])

        prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )

        try:
            result = self.llm.invoke(prompt)
            answer = result.strip()

        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = "Sorry, I encountered an error generating the answer."

        return answer

    def generate_answer_with_citations(self, query, search_results, chat_history=None):
        if not search_results:
             return {
                'answer': "I cannot find relevant information in the document.",
                'citations': [],
                'context_used': 0
            }

        # --- RERANKING STEP ---
        print(f"Reranking {len(search_results)} candidates...")
        
        pass_chunks = [result['chunk'] for result in search_results]
        cross_inp = [[query, chunk['content']] for chunk in pass_chunks]
        
        cross_scores = self.cross_encoder.predict(cross_inp)
        
        # Combine chunk with new score
        reranked_results = []
        for i, score in enumerate(cross_scores):
            reranked_results.append({
                'chunk': pass_chunks[i],
                'score': float(score), # Convert numpy float to python float
                'rank': i + 1 # Use current index as temp rank
            })
            
        # Sort by Cross-Encoder score (High is better)
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take Top-3 most relevant after reranking
        top_k_results = reranked_results[:3]
        
        context_chunks = [result['chunk'] for result in top_k_results]

        # Use the original query (or rephrased one passed in) for generation
        answer = self.generate_answer(query, context_chunks)

        citations = []
        for i, result in enumerate(top_k_results):
            chunk = result['chunk']
            citations.append({
                'rank': i + 1,
                'source': chunk['source'],
                'page': chunk['page'],
                'type': chunk['type'],
                'relevance_score': result['score']
            })

        return {
            'answer': answer,
            'citations': citations,
            'context_used': len(context_chunks)
        }

    def generate_summary(self, context_chunks):
        # Concatenate top chunks for summary, keeping within reasonable token limits
        # We take a broader sample of chunks to provide a better overall summary
        context_text = ""
        for i in range(0, min(len(context_chunks), 20), 4): # Sample across the chunks
            chunk = context_chunks[i]
            if len(context_text) + len(chunk['content']) < 2000:
                context_text += chunk['content'] + "\n\n"
            else:
                break

        prompt = f"Provide a brief overview of the document based on these sections:\n\n{context_text[:2000]}\n\nOverview:"

        try:
            result = self.llm.invoke(prompt)
            summary = result.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            summary = "Sorry, I encountered an error generating the summary."

        return summary

class SimpleQA:
    def __init__(self):
        print()

    def generate_answer_with_citations(self, query, search_results):
        if not search_results:
            return {
                'answer': "No relevant information found in the document.",
                'citations': [],
                'context_used': 0
            }
        top_chunks = search_results[:3]

        answer_parts = []
        for result in top_chunks:
            chunk = result['chunk']
            snippet = chunk['content'][:200].strip()
            if snippet:
                answer_parts.append(f"From {chunk['source']}: {snippet}...")

        answer = "\n\n".join(answer_parts) if answer_parts else "No relevant information found."

        citations = []
        for i, result in enumerate(top_chunks):
            chunk = result['chunk']
            citations.append({
                'rank': i + 1,
                'source': chunk['source'],
                'page': chunk['page'],
                'type': chunk['type'],
                'relevance_score': result['score']
            })

        return {
            'answer': answer,
            'citations': citations,
            'context_used': len(search_results)
        }

if __name__ == "__main__":

    test_results = [
        {
            'chunk': {
                'content': 'Qatar economy grew by 5% in 2024 driven by strong non-hydrocarbon sector growth.',
                'page': 1,
                'type': 'text',
                'source': 'Page 1'
            },
            'score': 0.85
        },
        {
            'chunk': {
                'content': 'The banking sector remains healthy with strong capital ratios.',
                'page': 2,
                'type': 'text',
                'source': 'Page 2'
            },
            'score': 0.72
        }
    ]
    try:
        print("\n1. Test ")
        qa = LLMQA()
        result = qa.generate_answer_with_citations("What is Qatar's growth?", test_results)
        print(f"\nAnswer: {result['answer']}")
        print(f"Citations: {len(result['citations'])} sources")

    except Exception as e:
        print(f"\nLangChain LLMQA failed: {e}")
        print("\n2. Test Fallback ")
        qa = SimpleQA()
        result = qa.generate_answer_with_citations("What is Qatar's growth?", test_results)
        print(f"\nAnswer: {result['answer']}")
        print(f"Citations: {len(result['citations'])} sources")