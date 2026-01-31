import time
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import CrossEncoder
import torch
import logging
from modules import config

logger = logging.getLogger(__name__)

class LLMQA:
    """
    LLM-based Question Answering system with cross-encoder reranking.
    
    Features:
    - HuggingFace transformer models via LangChain
    - Cross-encoder reranking for improved relevance
    - Citation generation with source tracking
    """
    
    def __init__(self, model_name='google/flan-t5-base'):
        """
        Initialize LLM QA system.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        logger.info(f"Loading LLM model via LangChain: {model_name}")

        device = 0 if torch.cuda.is_available() else -1
        device_name = 'GPU' if device == 0 else 'CPU'

        try:
            # Load Cross Encoder for Reranking
            logger.info("Loading CrossEncoder for Reranking...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.MAX_NEW_TOKENS,
                device=device,
                do_sample=True,
                temperature=config.TEMPERATURE,
                repetition_penalty=config.REPETITION_PENALTY
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

            logger.info(f"LangChain LLM loaded on {device_name}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
            logger.debug(f"Original: {query} -> Rephrased: {standalone_question}")
            return standalone_question
        except Exception as e:
            logger.error(f"Error rephrasing question: {e}")
            return query

    def generate_answer(self, query, context_chunks):
        """Generate answer from context chunks with length optimization."""
        # Limit context length for faster processing
        context_parts = []
        total_length = 0
        
        for chunk in context_chunks:
            chunk_text = f"[Source: {chunk['source']}]\n{chunk['content']}"
            if total_length + len(chunk_text) > config.MAX_CONTEXT_LENGTH:
                # Truncate the chunk to fit within limit
                remaining_space = config.MAX_CONTEXT_LENGTH - total_length
                if remaining_space > 100:  # Only add if meaningful space left
                    truncated_chunk = chunk_text[:remaining_space-3] + "..."
                    context_parts.append(truncated_chunk)
                break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        context_text = "\n\n".join(context_parts)

        prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )

        try:
            result = self.llm.invoke(prompt)
            answer = result.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = "Sorry, I encountered an error generating the answer."

        return answer

    def generate_answer_with_citations(self, query, search_results, chat_history=None):
        """
        Generate answer with citations using reranking for improved relevance.
        
        Args:
            query: User question
            search_results: Results from vector store search
            chat_history: Previous conversation context
            
        Returns:
            Dict with answer, citations, and metadata
        """
        if not search_results:
             return {
                'answer': "I cannot find relevant information in the document.",
                'citations': [],
                'context_used': 0
            }

        # Rerank results using cross-encoder
        logger.info(f"Reranking {len(search_results)} candidates...")
        
        try:
            chunks = [result['chunk'] for result in search_results]
            cross_inputs = [[query, chunk['content']] for chunk in chunks]
            
            cross_scores = self.cross_encoder.predict(cross_inputs)
            
            # Combine chunks with reranking scores
            reranked_results = []
            for i, score in enumerate(cross_scores):
                reranked_results.append({
                    'chunk': chunks[i],
                    'score': float(score),
                    'original_rank': i + 1,
                    'original_score': search_results[i]['score']
                })
                
            # Sort by cross-encoder score (higher is better)
            reranked_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Take top results for context
            top_k_results = reranked_results[:config.RERANK_TOP_K]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original search results
            '''top_k_results = search_results[:config.RERANK_TOP_K]
            for result in top_k_results:
                result['score'] = result.get('score', 0.0)'''
        
        context_chunks = [result['chunk'] for result in top_k_results]

        # Generate answer
        answer = self.generate_answer(query, context_chunks)

        # Create citations
        citations = []
        for i, result in enumerate(top_k_results):
            chunk = result['chunk']
            citations.append({
                'rank': i + 1,
                'source': chunk['source'],
                'page': chunk['page'],
                'type': chunk['type'],
                'relevance_score': result['score'],
                'content_preview': chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
            })

        return {
            'answer': answer,
            'citations': citations,
            'context_used': len(context_chunks),
            'total_candidates': len(search_results)
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
            logger.error(f"Error generating summary: {e}")
            summary = "Sorry, I encountered an error generating the summary."

        return summary

class SimpleQA:
    """Fallback QA system when LLM loading fails."""
    
    def __init__(self):
        logger.info("Initializing SimpleQA (Fallback)")

    def generate_answer_with_citations(self, query, search_results):
        """
        Generate simple answer by combining top search results.
        
        Args:
            query: User question
            search_results: Results from vector store search
            
        Returns:
            Dict with answer and citations
        """
        if not search_results:
            return {
                'answer': "No relevant information found in the document.",
                'citations': [],
                'context_used': 0
            }
            
        # Take top 3 results
        top_chunks = search_results[:3]

        # Create a more structured answer
        answer_parts = []
        answer_parts.append(f"Based on the document, here's what I found regarding '{query}':\n")
        
        for i, result in enumerate(top_chunks, 1):
            chunk = result['chunk']
            snippet = chunk['content'][:300].strip()
            if snippet:
                answer_parts.append(f"{i}. From {chunk['source']}: {snippet}...")

        answer = "\n\n".join(answer_parts) if len(answer_parts) > 1 else "No relevant information found."

        # Create citations
        citations = []
        for i, result in enumerate(top_chunks):
            chunk = result['chunk']
            citations.append({
                'rank': i + 1,
                'source': chunk['source'],
                'page': chunk['page'],
                'type': chunk['type'],
                'relevance_score': result.get('score', 0.0),
                'content_preview': chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
            })

        return {
            'answer': answer,
            'citations': citations,
            'context_used': len(top_chunks),
            'fallback_used': True
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Setup logging for standalone run
    
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
