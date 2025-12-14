"""
LLM Evaluation Pipeline for RAG Systems
Evaluates AI responses on three levels: Simple (heuristic), Medium (semantic), Robust (LLM-based)
Implements tiered sampling strategy for scalability.
"""

import json
import random
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import Counter
import re

try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Some features may be limited.")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Level 2 metrics will be skipped.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some metrics may be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Some array operations may be limited.")

try:
    import os
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not available. Install with: pip install openai")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline sampling rates."""
    level1_sample_rate: float = 1.0  # 100% - Always run
    level2_sample_rate: float = 0.10  # 10% - Sample
    level3_sample_rate: float = 0.01  # 1% - Tiny sample
    random_seed: int = 42
    openai_api_key: Optional[str] = None  # OpenAI API key for Level 3 evaluations
    openai_model: str = "gpt-3.5-turbo"  # OpenAI model to use


@dataclass
class EvaluationResult:
    """Structure for evaluation results."""
    message_id: Optional[str]
    user_query: str
    ai_response: str
    metrics: Dict[str, Any]
    retrieval_metrics: Optional[Dict[str, Any]] = None
    operational_metrics: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None  # Deprecated: use operational_metrics.latency_ms
    cost_estimate: Optional[float] = None  # Deprecated: use operational_metrics.cost_usd


class DataMapper:
    """Handles mapping between conversation turns and context vectors."""
    
    @staticmethod
    def load_json(file_path: str) -> Dict:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def extract_final_response(context_vectors: Dict) -> Optional[str]:
        """Extract final_response from context_vectors.json."""
        try:
            final_response = context_vectors['data']['sources']['final_response']
            if isinstance(final_response, list):
                return ' '.join(final_response)
            return str(final_response)
        except KeyError:
            return None
    
    @staticmethod
    def extract_context_chunks(context_vectors: Dict) -> List[str]:
        """Extract context chunks from vector_data that are in vectors_used."""
        try:
            vector_data = context_vectors['data']['vector_data']
            vectors_used = context_vectors['data']['sources'].get('vectors_used', [])
            
            # Convert vectors_used to set for faster lookup
            vectors_used_set = set(vectors_used) if vectors_used else set()
            
            # If vectors_used is empty, return all chunks (fallback behavior)
            if not vectors_used_set:
                chunks = [item.get('text', '') for item in vector_data if item.get('text')]
                return chunks
            
            # Only extract chunks whose id is in vectors_used
            chunks = [
                item.get('text', '') 
                for item in vector_data 
                if item.get('text') and item.get('id') in vectors_used_set
            ]
            
            return chunks
        except KeyError:
            return []
    
    @staticmethod
    def extract_vectors_used(context_vectors: Dict) -> List[int]:
        """Extract vectors_used IDs from context_vectors.json."""
        try:
            vectors_used = context_vectors['data']['sources'].get('vectors_used', [])
            # Ensure all are integers
            return [int(v) for v in vectors_used if v is not None]
        except (KeyError, ValueError, TypeError):
            return []
    
    @staticmethod
    def extract_vectors_info(context_vectors: Dict) -> List[Dict[str, Any]]:
        """Extract vectors_info (ranking/scores) from context_vectors.json."""
        try:
            vectors_info = context_vectors['data']['sources'].get('vectors_info', [])
            vector_ids_list = context_vectors['data']['sources'].get('vector_ids', [])
            
            # Convert vector_ids_list to integers for easier matching
            vector_ids_int = []
            for vid in vector_ids_list:
                try:
                    vector_ids_int.append(int(vid))
                except (ValueError, TypeError):
                    pass
            
            # Map vector_id in vectors_info to actual content IDs
            # Strategy: vector_id might be an index or a direct ID
            mapped_vectors_info = []
            for idx, item in enumerate(vectors_info):
                vector_id = item.get('vector_id')
                mapped_item = item.copy()
                content_id = None
                
                if vector_id is not None:
                    try:
                        vector_id_int = int(vector_id)
                        # Strategy 1: If vector_id matches a content ID directly
                        if vector_id_int in vector_ids_int:
                            content_id = vector_id_int
                        # Strategy 2: If vector_id is an index into vector_ids_list
                        elif 0 <= vector_id_int < len(vector_ids_int):
                            content_id = vector_ids_int[vector_id_int]
                        # Strategy 3: Use position-based mapping (idx)
                        elif idx < len(vector_ids_int):
                            content_id = vector_ids_int[idx]
                        else:
                            # Fallback: assume vector_id is the content ID
                            content_id = vector_id_int
                    except (ValueError, TypeError):
                        # Try string-based matching
                        if str(vector_id) in [str(vid) for vid in vector_ids_list]:
                            try:
                                content_id = int(vector_id)
                            except (ValueError, TypeError):
                                pass
                        # Try position-based if idx is valid
                        elif idx < len(vector_ids_int):
                            content_id = vector_ids_int[idx]
                
                mapped_item['content_id'] = content_id
                mapped_vectors_info.append(mapped_item)
            
            # Sort by score descending (highest first) if not already sorted
            if mapped_vectors_info:
                mapped_vectors_info = sorted(mapped_vectors_info, key=lambda x: x.get('score', 0), reverse=True)
            
            return mapped_vectors_info
        except (KeyError, TypeError) as e:
            print(f"Error extracting vectors_info: {e}")
            return []
    
    @staticmethod
    def extract_all_context_chunks(context_vectors: Dict) -> Dict[int, str]:
        """Extract all context chunks mapped by their ID for retrieval evaluation."""
        try:
            vector_data = context_vectors['data']['vector_data']
            id_to_text = {}
            for item in vector_data:
                vector_id = item.get('id')
                text = item.get('text', '')
                if vector_id is not None and text:
                    id_to_text[int(vector_id)] = text
            return id_to_text
        except (KeyError, ValueError, TypeError):
            return {}
    
    @staticmethod
    def find_matching_turn(conversation: Dict, final_response: str) -> Optional[Tuple[int, Dict]]:
        """
        Find the AI response turn that matches final_response.
        Returns (turn_number, turn_dict) or None.
        """
        turns = conversation.get('conversation_turns', [])
        
        # Normalize final_response for comparison
        final_response_normalized = DataMapper._normalize_text(final_response)
        
        for turn in turns:
            if turn.get('role') == 'AI/Chatbot':
                message = turn.get('message', '')
                message_normalized = DataMapper._normalize_text(message)
                
                # Check if responses match (allowing for minor formatting differences)
                if DataMapper._texts_match(final_response_normalized, message_normalized):
                    return turn.get('turn'), turn
        
        return None
    
    @staticmethod
    def get_user_query_for_turn(conversation: Dict, ai_turn_number: int) -> Optional[str]:
        """Get the user query that triggered the AI response (turn N-1)."""
        turns = conversation.get('conversation_turns', [])
        
        for turn in turns:
            if turn.get('turn') == ai_turn_number - 1 and turn.get('role') == 'User':
                return turn.get('message', '')
        
        return None
    
    @staticmethod
    def get_last_user_query(conversation: Dict) -> Optional[str]:
        """Get the last user query from the conversation as fallback."""
        turns = conversation.get('conversation_turns', [])
        
        # Iterate backwards to find the last user message
        for turn in reversed(turns):
            if turn.get('role') == 'User':
                return turn.get('message', '')
        
        return None
    
    @staticmethod
    def get_user_turn(conversation: Dict, ai_turn_number: int) -> Optional[Dict]:
        """Get the user turn dictionary that triggered the AI response (turn N-1)."""
        turns = conversation.get('conversation_turns', [])
        
        for turn in turns:
            if turn.get('turn') == ai_turn_number - 1 and turn.get('role') == 'User':
                return turn
        
        return None
    
    @staticmethod
    def get_ai_turn(conversation: Dict, turn_number: int) -> Optional[Dict]:
        """Get the AI turn dictionary by turn number."""
        turns = conversation.get('conversation_turns', [])
        
        for turn in turns:
            if turn.get('turn') == turn_number and turn.get('role') in ['AI', 'AI/Chatbot', 'Chatbot']:
                return turn
        
        return None
    
    @staticmethod
    def get_last_user_turn(conversation: Dict) -> Optional[Dict]:
        """Get the last user turn dictionary as fallback."""
        turns = conversation.get('conversation_turns', [])
        
        # Iterate backwards to find the last user message
        for turn in reversed(turns):
            if turn.get('role') == 'User':
                return turn
        
        return None
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        # Remove markdown links, extra whitespace
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove markdown links
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    @staticmethod
    def _texts_match(text1: str, text2: str, threshold: float = 0.85) -> bool:
        """Check if two texts match (with fuzzy matching)."""
        # Simple token-based similarity
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        if not tokens1 or not tokens2:
            return False
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        jaccard = intersection / union if union > 0 else 0
        return jaccard >= threshold
    
    @staticmethod
    def map_data(conversation_path: str, context_vectors_path: str) -> List[Dict]:
        """
        Map context vectors to conversation turns.
        Returns list of tuples: (user_query, ai_response, context_chunks, message_id)
        Falls back to last user query if mapping fails.
        """
        conversation = DataMapper.load_json(conversation_path)
        context_vectors = DataMapper.load_json(context_vectors_path)
        
        final_response = DataMapper.extract_final_response(context_vectors)
        if not final_response:
            return []
        
        context_chunks = DataMapper.extract_context_chunks(context_vectors)
        message_id = context_vectors.get('data', {}).get('sources', {}).get('message_id')
        
        # Try to find matching turn
        matching_turn = DataMapper.find_matching_turn(conversation, final_response)
        
        if matching_turn:
            # Found matching turn - use the preceding user query
            turn_number, turn_dict = matching_turn
            user_query = DataMapper.get_user_query_for_turn(conversation, turn_number)
            
            if user_query:
                # Get turn dictionaries for operational metrics
                user_turn = DataMapper.get_user_turn(conversation, turn_number)
                ai_turn = DataMapper.get_ai_turn(conversation, turn_number)
                
                return [{
                    'user_query': user_query,
                    'ai_response': final_response,
                    'context_chunks': context_chunks,
                    'message_id': message_id,
                    'turn_number': turn_number,
                    'mapping_method': 'exact_match',
                    'vectors_used': DataMapper.extract_vectors_used(context_vectors),
                    'vectors_info': DataMapper.extract_vectors_info(context_vectors),
                    'all_chunks_map': DataMapper.extract_all_context_chunks(context_vectors),
                    'user_turn': user_turn,
                    'ai_turn': ai_turn,
                    'vectors_data': context_vectors.get('data', {}).get('vector_data', [])
                }]
        
        # Fallback: Use last user query from conversation
        last_user_query = DataMapper.get_last_user_query(conversation)
        last_user_turn = DataMapper.get_last_user_turn(conversation)
        
        if last_user_query:
            print(f"Warning: Could not find exact match for response. Using last user query as fallback.")
            print(f"Last user query: {last_user_query}")
            print(f"Final response: {final_response}")
            
            # Try to find the AI turn that matches final_response
            ai_turn = None
            turns = conversation.get('conversation_turns', [])
            for turn in reversed(turns):
                if turn.get('role') in ['AI', 'AI/Chatbot', 'Chatbot']:
                    # Check if this turn's message matches final_response
                    if DataMapper._texts_match(final_response, turn.get('message', '')):
                        ai_turn = turn
                        break
            
            return [{
                'user_query': last_user_query,
                'ai_response': final_response,
                'context_chunks': context_chunks,
                'message_id': message_id,
                'turn_number': None,
                'mapping_method': 'fallback_last_user_query',
                'vectors_used': DataMapper.extract_vectors_used(context_vectors),
                'vectors_info': DataMapper.extract_vectors_info(context_vectors),
                'all_chunks_map': DataMapper.extract_all_context_chunks(context_vectors),
                'user_turn': last_user_turn,
                'ai_turn': ai_turn,
                'vectors_data': context_vectors.get('data', {}).get('vector_data', [])
            }]
        
        # If even fallback fails, return empty
        print(f"Warning: Could not find any user query in conversation. Skipping evaluation.")
        return []


class OperationalMetricsCalculator:
    """Calculates operational metrics: latency, cost, and retrieval efficiency."""
    
    # Pricing constants (GPT-4o pricing)
    INPUT_PRICE_PER_1K = 0.005  # $0.005 per 1K input tokens
    OUTPUT_PRICE_PER_1K = 0.015  # $0.015 per 1K output tokens
    MAX_REASONABLE_LATENCY_MS = 5 * 60 * 1000  # 5 minutes in milliseconds
    
    @staticmethod
    def calculate_inferred_latency(user_turn: Optional[Dict], ai_turn: Optional[Dict]) -> Optional[float]:
        """
        Calculate latency from timestamps.
        Returns latency in milliseconds, or None if invalid.
        """
        if not user_turn or not ai_turn:
            return None
        
        user_timestamp_str = user_turn.get('created_at')
        ai_timestamp_str = ai_turn.get('created_at')
        
        if not user_timestamp_str or not ai_timestamp_str:
            return None
        
        try:
            # Parse ISO 8601 timestamps
            user_timestamp = datetime.fromisoformat(user_timestamp_str.replace('Z', '+00:00'))
            ai_timestamp = datetime.fromisoformat(ai_timestamp_str.replace('Z', '+00:00'))
            
            # Calculate difference in milliseconds
            time_diff = (ai_timestamp - user_timestamp).total_seconds() * 1000
            
            # Edge case: negative or unreasonably large (session break)
            if time_diff < 0 or time_diff > OperationalMetricsCalculator.MAX_REASONABLE_LATENCY_MS:
                return None
            
            return round(time_diff, 2)
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse timestamps for latency calculation: {e}")
            return None
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count from text (rough approximation: 1 token ≈ 4 characters)."""
        return max(1, len(text) // 4)
    
    @staticmethod
    def calculate_estimated_cost(
        query: str,
        response: str,
        vectors_data: List[Dict],
        vectors_used_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Calculate estimated cost based on token counts.
        Returns dict with cost breakdown and token counts.
        """
        # Convert vectors_used_ids to set for faster lookup
        vectors_used_set = set(vectors_used_ids) if vectors_used_ids else set()
        
        # Calculate context tokens (only for vectors that were used)
        context_tokens = 0
        for vector in vectors_data:
            vector_id = vector.get('id')
            if vector_id is not None and int(vector_id) in vectors_used_set:
                tokens = vector.get('tokens', 0)
                if isinstance(tokens, (int, float)):
                    context_tokens += int(tokens)
        
        # Estimate query tokens
        query_tokens = OperationalMetricsCalculator.estimate_tokens(query)
        
        # Estimate response tokens
        response_tokens = OperationalMetricsCalculator.estimate_tokens(response)
        
        # Calculate costs
        input_tokens = context_tokens + query_tokens
        output_tokens = response_tokens
        total_tokens = input_tokens + output_tokens
        
        input_cost = (input_tokens / 1000) * OperationalMetricsCalculator.INPUT_PRICE_PER_1K
        output_cost = (output_tokens / 1000) * OperationalMetricsCalculator.OUTPUT_PRICE_PER_1K
        total_cost = input_cost + output_cost
        
        return {
            'cost_usd': round(total_cost, 6),
            'input_cost_usd': round(input_cost, 6),
            'output_cost_usd': round(output_cost, 6),
            'token_counts': {
                'prompt_tokens': input_tokens,
                'completion_tokens': output_tokens,
                'total_tokens': total_tokens,
                'context_tokens': context_tokens,
                'query_tokens': query_tokens,
                'response_tokens': response_tokens
            }
        }
    
    @staticmethod
    def calculate_retrieval_efficiency(
        vectors_data: List[Dict],
        vectors_used_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Calculate retrieval efficiency (waste ratio).
        Returns dict with efficiency metrics.
        """
        if not vectors_data:
            return {'waste_ratio': 0.0, 'total_retrieved_tokens': 0, 'used_tokens': 0}
        
        # Convert vectors_used_ids to set for faster lookup
        vectors_used_set = set(vectors_used_ids) if vectors_used_ids else set()
        
        # Calculate total retrieved tokens (all vectors)
        total_retrieved_tokens = 0
        for vector in vectors_data:
            tokens = vector.get('tokens', 0)
            if isinstance(tokens, (int, float)):
                total_retrieved_tokens += int(tokens)
        
        # Calculate used tokens (only vectors in vectors_used)
        used_tokens = 0
        for vector in vectors_data:
            vector_id = vector.get('id')
            if vector_id is not None and int(vector_id) in vectors_used_set:
                tokens = vector.get('tokens', 0)
                if isinstance(tokens, (int, float)):
                    used_tokens += int(tokens)
        
        # Calculate waste ratio
        if total_retrieved_tokens == 0:
            waste_ratio = 0.0
        else:
            waste_ratio = 1.0 - (used_tokens / total_retrieved_tokens)
        
        return {
            'waste_ratio': round(max(0.0, min(1.0, waste_ratio)), 4),
            'total_retrieved_tokens': total_retrieved_tokens,
            'used_tokens': used_tokens,
            'retrieved_count': len(vectors_data),
            'used_count': len(vectors_used_ids) if vectors_used_ids else 0
        }
    
    @staticmethod
    def calculate_operational_metrics(
        user_query: str,
        ai_response: str,
        user_turn: Optional[Dict],
        ai_turn: Optional[Dict],
        vectors_data: List[Dict],
        vectors_used_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Calculate all operational metrics.
        Returns dict with latency, cost, and retrieval efficiency.
        """
        # Calculate latency
        latency_ms = OperationalMetricsCalculator.calculate_inferred_latency(user_turn, ai_turn)
        
        # Calculate cost
        cost_metrics = OperationalMetricsCalculator.calculate_estimated_cost(
            user_query, ai_response, vectors_data, vectors_used_ids
        )
        
        # Calculate retrieval efficiency
        efficiency_metrics = OperationalMetricsCalculator.calculate_retrieval_efficiency(
            vectors_data, vectors_used_ids
        )
        
        return {
            'latency_ms': latency_ms,
            'cost_usd': cost_metrics['cost_usd'],
            'token_counts': cost_metrics['token_counts'],
            'retrieval_efficiency': efficiency_metrics
        }


class Level1Evaluator:
    """Level 1: Simple heuristic/lexical metrics (Fast & Cheap)."""
    
    @staticmethod
    def calculate_rouge_l(response: str, context_chunks: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE-L (Longest Common Subsequence) between response and context.
        Returns recall, precision, f1.
        """
        if not response or not context_chunks:
            return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
        
        # Combine all context chunks
        combined_context = ' '.join(context_chunks)
        
        # Tokenize
        if NLTK_AVAILABLE:
            try:
                response_tokens = word_tokenize(response.lower())
                context_tokens = word_tokenize(combined_context.lower())
            except LookupError:
                # Fallback if NLTK resources not available
                response_tokens = response.lower().split()
                context_tokens = combined_context.lower().split()
        else:
            response_tokens = response.lower().split()
            context_tokens = combined_context.lower().split()
        
        # Calculate LCS
        lcs_length = Level1Evaluator._lcs_length(response_tokens, context_tokens)
        
        recall = lcs_length / len(response_tokens) if response_tokens else 0.0
        precision = lcs_length / len(context_tokens) if context_tokens else 0.0
        f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
        
        return {
            'recall': round(recall, 4),
            'precision': round(precision, 4),
            'f1': round(f1, 4)
        }
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of Longest Common Subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def calculate_jaccard_similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity (token overlap) between two texts."""
        if not text1 or not text2:
            return 0.0
        
        if NLTK_AVAILABLE:
            try:
                tokens1 = set(word_tokenize(text1.lower()))
                tokens2 = set(word_tokenize(text2.lower()))
            except LookupError:
                # Fallback if NLTK resources not available
                tokens1 = set(text1.lower().split())
                tokens2 = set(text2.lower().split())
        else:
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
        
        # Remove stopwords for better relevance
        if NLTK_AVAILABLE:
            try:
                stop_words = set(stopwords.words('english'))
                tokens1 = tokens1 - stop_words
                tokens2 = tokens2 - stop_words
            except (LookupError, AttributeError):
                pass
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return round(intersection / union, 4) if union > 0 else 0.0
    
    @staticmethod
    def calculate_length_ratio_completeness(user_query: str, ai_response: str, threshold: float = 0.2) -> Dict[str, Any]:
        """
        Calculate completeness using Length/Information Ratio.
        Logic: len(Response) / len(Query). Flag as 0 if ratio < threshold.
        """
        if not user_query or not ai_response:
            return {'ratio': 0.0, 'score': 0.0, 'flagged': True}
        
        query_length = len(user_query.strip())
        response_length = len(ai_response.strip())
        
        if query_length == 0:
            return {'ratio': 0.0, 'score': 0.0, 'flagged': True}
        
        ratio = response_length / query_length
        flagged = ratio < threshold
        
        # Score: 0 if flagged, otherwise normalize to 0-1 (capped at 1.0)
        score = 0.0 if flagged else min(1.0, ratio)
        
        return {
            'ratio': round(ratio, 4),
            'score': round(score, 4),
            'flagged': flagged,
            'threshold': threshold
        }
    
    @staticmethod
    def evaluate(user_query: str, ai_response: str, context_chunks: List[str]) -> Dict[str, Any]:
        """Run Level 1 evaluation."""
        factuality = Level1Evaluator.calculate_rouge_l(ai_response, context_chunks)
        relevance = Level1Evaluator.calculate_jaccard_similarity(user_query, ai_response)
        completeness = Level1Evaluator.calculate_length_ratio_completeness(user_query, ai_response)
        
        return {
            'level1': {
                'factuality': factuality,
                'relevance': {
                    'simple': relevance
                },
                'completeness': {
                    'simple': completeness
                }
            }
        }


class Level2Evaluator:
    """Level 2: Semantic metrics using local models (Medium complexity)."""
    
    def __init__(self):
        self.cross_encoder = None
        self.sentence_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize semantic models if available."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Cross-encoder for factuality (NLI)
                self.cross_encoder = CrossEncoder('cross-encoder/nli-deberta-v3-small')
                # Sentence transformer for relevance
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not initialize Level 2 models: {e}")
                self.cross_encoder = None
                self.sentence_model = None
    
    def calculate_entailment_score(self, context_chunks: List[str], response: str) -> float:
        """
        Calculate entailment score using cross-encoder.
        Returns probability that context implies response.
        """
        if not self.cross_encoder or not context_chunks or not response:
            return 0.0
        
        # Combine context chunks
        combined_context = ' '.join(context_chunks)
        
        # Truncate if too long
        if len(combined_context) > 512:
            combined_context = combined_context[:512]
        if len(response) > 512:
            response = response[:512]
        
        try:
            # Cross-encoder expects [premise, hypothesis]
            scores = self.cross_encoder.predict([[combined_context, response]])
            
            # Helper function to extract scalar from numpy array or nested structures
            def extract_scalar(value):
                """Extract scalar value from numpy array or nested structures."""
                if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    # Flatten multi-dimensional arrays
                    if value.ndim > 0:
                        value = value.flatten()[0]
                    # Convert numpy scalar to Python float
                    return float(value.item() if hasattr(value, 'item') else value)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    # Handle lists/tuples - get first element
                    if len(value) > 0:
                        return extract_scalar(value[0])
                    return 0.0
                else:
                    # Already a scalar
                    return float(value)
            
            # Handle different output formats
            # Cross-encoder typically returns a 2D array: [[contradiction, neutral, entailment]]
            # or sometimes just [contradiction, neutral, entailment]
            
            # Convert to list if numpy array
            if NUMPY_AVAILABLE and isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            # Extract the score array
            if isinstance(scores, (list, tuple)) and len(scores) > 0:
                score_array = scores[0] if isinstance(scores[0], (list, tuple)) else scores
            else:
                score_array = scores
            
            # Convert numpy elements to Python floats
            if isinstance(score_array, (list, tuple)):
                score_array = [extract_scalar(x) for x in score_array]
            else:
                score_array = [extract_scalar(score_array)]
            
            # Determine which score to use
            if len(score_array) >= 3:
                # [contradiction, neutral, entailment] - use entailment (index 2)
                entailment_score = score_array[2]
            elif len(score_array) == 2:
                # Binary classification - use second score
                entailment_score = score_array[1]
            else:
                # Single score
                entailment_score = score_array[0] if score_array else 0.0
            
            # Normalize to [0, 1] range (scores might be logits)
            if entailment_score < 0:
                entailment_score = 0.0
            elif entailment_score > 1:
                # If it's a logit, apply sigmoid approximation
                entailment_score = 1 / (1 + abs(entailment_score))
            
            return round(max(0.0, min(1.0, entailment_score)), 4)
        except Exception as e:
            print(f"Error in entailment calculation: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between embeddings."""
        if not self.sentence_model or not text1 or not text2:
            return 0.0
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return round(float(similarity), 4)
        except Exception as e:
            print(f"Error in semantic similarity calculation: {e}")
            return 0.0
    
    @staticmethod
    def check_instruction_following(user_query: str, ai_response: str) -> Dict[str, Any]:
        """
        IFEval Methodology: Verifiable Instruction Checker.
        Detects explicit constraints in user query and checks if response adheres to them.
        """
        if not user_query or not ai_response:
            return {'score': 1.0, 'constraints_found': 0, 'constraints_met': 0}
        
        query_lower = user_query.lower()
        response_lower = ai_response.lower()
        
        # Define constraint patterns
        constraints = []
        
        # Check for bullet points / list format
        if any(phrase in query_lower for phrase in ['bullet point', 'bullet points', 'bulleted list', 'list format']):
            constraints.append({
                'type': 'bullet_points',
                'required': True,
                'met': bool(re.search(r'^[\s]*[-•*]\s', response_lower, re.MULTILINE) or 
                           re.search(r'^\d+[\.\)]\s', response_lower, re.MULTILINE))
            })
        
        # Check for JSON format
        if 'json' in query_lower:
            constraints.append({
                'type': 'json',
                'required': True,
                'met': bool(re.search(r'\{[\s\S]*\}', ai_response) or 'json' in response_lower)
            })
        
        # Check for "no commas" constraint
        if 'no comma' in query_lower or 'without comma' in query_lower:
            constraints.append({
                'type': 'no_commas',
                'required': True,
                'met': ',' not in ai_response
            })
        
        # Check for word count constraints
        word_count_match = re.search(r'(\d+)\s*(words?|word\s+limit)', query_lower)
        if word_count_match:
            max_words = int(word_count_match.group(1))
            response_word_count = len(ai_response.split())
            constraints.append({
                'type': 'word_count',
                'required': True,
                'met': response_word_count <= max_words,
                'actual': response_word_count,
                'max': max_words
            })
        
        # Check for "in one sentence" or "briefly"
        if any(phrase in query_lower for phrase in ['one sentence', 'single sentence', 'briefly', 'brief']):
            sentence_count = len(re.split(r'[.!?]+', ai_response))
            constraints.append({
                'type': 'single_sentence',
                'required': True,
                'met': sentence_count <= 2  # Allow for minor variations
            })
        
        # Calculate score
        if not constraints:
            return {'score': 1.0, 'constraints_found': 0, 'constraints_met': 0, 'details': []}
        
        constraints_met = sum(1 for c in constraints if c['met'])
        score = constraints_met / len(constraints) if constraints else 1.0
        
        return {
            'score': round(score, 4),
            'constraints_found': len(constraints),
            'constraints_met': constraints_met,
            'details': constraints
        }
    
    def evaluate(self, user_query: str, ai_response: str, context_chunks: List[str]) -> Optional[Dict[str, Any]]:
        """Run Level 2 evaluation."""
        if not self.cross_encoder or not self.sentence_model:
            return None
        
        factuality = self.calculate_entailment_score(context_chunks, ai_response)
        relevance = self.calculate_semantic_similarity(user_query, ai_response)
        completeness = Level2Evaluator.check_instruction_following(user_query, ai_response)
        
        return {
            'level2': {
                'factuality': factuality,
                'relevance': {
                    'semantic': relevance
                },
                'completeness': {
                    'ifeval': completeness
                }
            }
        }


class Level3Evaluator:
    """Level 3: Robust LLM-based metrics (FActScore logic)."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize Level3Evaluator with OpenAI API configuration."""
        self.config = config
        self.client = None
        self.api_key_available = False
        
        # Check for API key
        api_key = None
        if config and config.openai_api_key:
            api_key = config.openai_api_key
        elif OPENAI_AVAILABLE:
            # Try to get from environment variable
            api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=api_key)
                self.api_key_available = True
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                self.api_key_available = False
        else:
            if not OPENAI_AVAILABLE:
                print("Warning: OpenAI package not installed. Level 3 evaluation will not work.")
            elif not api_key:
                print("Warning: OpenAI API key not set. Set OPENAI_API_KEY environment variable or pass openai_api_key in config. Level 3 evaluation will not work.")
            self.api_key_available = False
    
    def _llm_call(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """
        Make actual OpenAI API call.
        Returns None if API key is not available.
        """
        if not self.api_key_available or not self.client:
            return None
        
        try:
            model_name = model or (self.config.openai_model if self.config else "gpt-3.5-turbo")
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent evaluation
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return None
    
    @staticmethod
    def decompose_into_atomic_facts(response: str) -> List[str]:
        """
        Decompose response into atomic facts (statements).
        Uses sentence tokenization and simple heuristics.
        """
        if not response:
            return []
        
        # Split into sentences
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(response)
            except LookupError:
                # Fallback if NLTK resources not available
                sentences = re.split(r'[.!?]+', response)
                sentences = [s.strip() for s in sentences if s.strip()]
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', response)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        atomic_facts = []
        for sentence in sentences:
            # Remove markdown and links
            sentence = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', sentence)
            sentence = sentence.strip()
            
            # Filter out very short sentences
            if len(sentence) > 10:
                atomic_facts.append(sentence)
        
        return atomic_facts
    
    def verify_fact_against_context(self, atomic_fact: str, context_chunks: List[str]) -> bool:
        """
        Verify if an atomic fact is supported by context chunks.
        Uses OpenAI API call.
        """
        if not self.api_key_available:
            return False
        
        combined_context = ' '.join(context_chunks[:5])  # Limit context
        
        prompt = f"""
        Context: {combined_context[:1000]}
        
        Statement: {atomic_fact}
        
        Is this statement supported by the context? Answer True or False.
        """
        
        result = self._llm_call(prompt)
        if result is None:
            return False
        return 'true' in result.lower()
    
    def calculate_factscore(self, response: str, context_chunks: List[str]) -> Optional[Dict[str, Any]]:
        """
        Calculate FActScore: fraction of atomic facts supported by context.
        Returns None if API key is not available.
        """
        if not self.api_key_available:
            return None
        
        atomic_facts = Level3Evaluator.decompose_into_atomic_facts(response)
        
        if not atomic_facts:
            return {
                'supported_facts': 0,
                'total_facts': 0,
                'score': 0.0
            }
        
        supported_count = 0
        for fact in atomic_facts:
            if self.verify_fact_against_context(fact, context_chunks):
                supported_count += 1
        
        score = supported_count / len(atomic_facts) if atomic_facts else 0.0
        
        return {
            'supported_facts': supported_count,
            'total_facts': len(atomic_facts),
            'score': round(score, 4)
        }
    
    def calculate_ragas_relevance(self, user_query: str, ai_response: str) -> Optional[Dict[str, Any]]:
        """
        Ragas Methodology: Reverse Question Generation.
        Generate a hypothetical question from the answer, then compare with original query.
        Returns None if API key is not available.
        """
        if not self.api_key_available:
            return None
        
        # Step 1: Generate hypothetical question from response
        prompt = f"""
        Given the following answer, generate a potential user question that this answer would respond to.
        
        Answer: {ai_response[:500]}
        
        Generate only the question, nothing else.
        """
        
        hypothetical_question = self._llm_call(prompt)
        if hypothetical_question is None:
            return None
        
        # Step 2: Calculate similarity between original query and hypothetical question
        # Use simple token-based similarity (in production, would use embeddings)
        if NLTK_AVAILABLE:
            try:
                orig_tokens = set(word_tokenize(user_query.lower()))
                hypo_tokens = set(word_tokenize(hypothetical_question.lower()))
            except LookupError:
                orig_tokens = set(user_query.lower().split())
                hypo_tokens = set(hypothetical_question.lower().split())
        else:
            orig_tokens = set(user_query.lower().split())
            hypo_tokens = set(hypothetical_question.lower().split())
        
        intersection = len(orig_tokens & hypo_tokens)
        union = len(orig_tokens | hypo_tokens)
        similarity = intersection / union if union > 0 else 0.0
        
        return {
            'score': round(similarity, 4),
            'hypothetical_question': hypothetical_question.strip(),
            'methodology': 'ragas_reverse_question_generation'
        }
    
    def calculate_rec_completeness(self, user_query: str, ai_response: str, context_chunks: List[str]) -> Optional[Dict[str, Any]]:
        """
        REC Framework: Rate, Explain, Cite.
        LLM evaluates completeness and cites missing details from context.
        Returns None if API key is not available.
        """
        if not self.api_key_available:
            return None
        
        combined_context = ' '.join(context_chunks[:5])[:2000]  # Limit context length
        
        prompt = f"""
        Evaluate the completeness of the AI response to the user query.
        
        User Query: {user_query}
        
        AI Response: {ai_response}
        
        Available Context: {combined_context[:1000]}
        
        Provide your evaluation in the following JSON format:
        {{
            "rating": <number from 1-5>,
            "explanation": "<brief explanation>",
            "missing_details": ["<detail1>", "<detail2>", ...],
            "cited_sources": ["<relevant context snippet>", ...]
        }}
        
        Rating scale:
        1 = Very incomplete, missing critical information
        2 = Incomplete, missing important details
        3 = Partially complete, some gaps
        4 = Mostly complete, minor gaps
        5 = Complete and comprehensive
        """
        
        result = self._llm_call(prompt)
        if result is None:
            return None
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                parsed = json.loads(json_match.group(0))
                
                rating = parsed.get('rating', 3)
                explanation = parsed.get('explanation', '')
                missing_details = parsed.get('missing_details', [])
                cited_sources = parsed.get('cited_sources', [])
                
                # Normalize rating to 0-1 scale
                normalized_score = (rating - 1) / 4.0
                
                return {
                    'score': round(normalized_score, 4),
                    'rating': rating,
                    'explanation': explanation,
                    'missing_details': missing_details,
                    'cited_sources': cited_sources,
                    'methodology': 'rec_framework'
                }
        except Exception as e:
            # Fallback parsing if JSON parsing fails
            # Extract rating number
            rating_match = re.search(r'"rating":\s*(\d+)', result)
            if rating_match:
                rating = int(rating_match.group(1))
                normalized_score = (rating - 1) / 4.0
            else:
                rating = 3
                normalized_score = 0.5
            
            return {
                'score': round(normalized_score, 4),
                'rating': rating,
                'explanation': 'Could not parse full response',
                'missing_details': [],
                'cited_sources': [],
                'methodology': 'rec_framework',
                'parse_error': str(e)
            }
    
    def evaluate(self, user_query: str, ai_response: str, context_chunks: List[str]) -> Optional[Dict[str, Any]]:
        """Run Level 3 evaluation. Returns None if API key is not available."""
        if not self.api_key_available:
            return None
        
        factuality = self.calculate_factscore(ai_response, context_chunks)
        relevance = self.calculate_ragas_relevance(user_query, ai_response)
        completeness = self.calculate_rec_completeness(user_query, ai_response, context_chunks)
        
        # If any critical metric failed, return None
        if factuality is None or relevance is None or completeness is None:
            return None
        
        return {
            'level3': {
                'factuality': factuality,
                'relevance': {
                    'ragas': relevance
                },
                'completeness': {
                    'rec': completeness
                }
            }
        }


class RetrievalEvaluator:
    """Evaluates retrieval and ranking effectiveness of vector database."""
    
    @staticmethod
    def calculate_hit_rate_at_k(vectors_info: List[Dict], vectors_used: List[int], k: int) -> float:
        """Calculate Hit Rate @ K: fraction of top K results that are in vectors_used."""
        if not vectors_info or not vectors_used or k <= 0:
            return 0.0
        
        vectors_used_set = set(vectors_used)
        top_k_ids = []
        
        for item in vectors_info[:k]:
            # Use content_id if available (from mapping), otherwise try vector_id
            content_id = item.get('content_id')
            if content_id is not None:
                top_k_ids.append(int(content_id))
            else:
                # Fallback to vector_id
                vector_id = item.get('vector_id')
                if vector_id is not None:
                    try:
                        vector_id_int = int(vector_id)
                        top_k_ids.append(vector_id_int)
                    except (ValueError, TypeError):
                        pass
        
        if not top_k_ids:
            return 0.0
        
        hits = sum(1 for vid in top_k_ids if vid in vectors_used_set)
        return round(hits / len(top_k_ids), 4) if top_k_ids else 0.0
    
    @staticmethod
    def calculate_mrr(vectors_info: List[Dict], vectors_used: List[int]) -> float:
        """Calculate Mean Reciprocal Rank: 1/rank of first relevant item."""
        if not vectors_info or not vectors_used:
            return 0.0
        
        vectors_used_set = set(vectors_used)
        
        for rank, item in enumerate(vectors_info, start=1):
            # Use content_id if available, otherwise try vector_id
            content_id = item.get('content_id')
            if content_id is not None:
                if int(content_id) in vectors_used_set:
                    return round(1.0 / rank, 4)
            else:
                vector_id = item.get('vector_id')
                if vector_id is not None:
                    try:
                        vector_id_int = int(vector_id)
                        if vector_id_int in vectors_used_set:
                            return round(1.0 / rank, 4)
                    except (ValueError, TypeError):
                        continue
        
        return 0.0
    
    @staticmethod
    def calculate_noise_ratio(vectors_info: List[Dict], vectors_used: List[int]) -> float:
        """Calculate Noise Ratio: 1 - (count_used / count_retrieved)."""
        if not vectors_info:
            return 0.0
        
        count_retrieved = len(vectors_info)
        count_used = len(vectors_used) if vectors_used else 0
        
        if count_retrieved == 0:
            return 0.0
        
        ratio = 1.0 - (count_used / count_retrieved)
        return round(max(0.0, min(1.0, ratio)), 4)
    
    @staticmethod
    def calculate_ndcg(vectors_info: List[Dict], vectors_used: List[int], k: Optional[int] = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        Treats vectors_used as relevance=1, others as relevance=0.
        """
        if not vectors_info or not vectors_used:
            return 0.0
        
        vectors_used_set = set(vectors_used)
        
        # Build relevance list (1 if in vectors_used, 0 otherwise)
        relevance_list = []
        for item in vectors_info:
            # Use content_id if available, otherwise try vector_id
            content_id = item.get('content_id')
            if content_id is not None:
                relevance = 1.0 if int(content_id) in vectors_used_set else 0.0
                relevance_list.append(relevance)
            else:
                vector_id = item.get('vector_id')
                if vector_id is not None:
                    try:
                        vector_id_int = int(vector_id)
                        relevance = 1.0 if vector_id_int in vectors_used_set else 0.0
                        relevance_list.append(relevance)
                    except (ValueError, TypeError):
                        relevance_list.append(0.0)
                else:
                    relevance_list.append(0.0)
        
        if not relevance_list:
            return 0.0
        
        # Limit to top k if specified
        if k is not None and k > 0:
            relevance_list = relevance_list[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_list, start=1):
            if rel > 0:
                dcg += rel / math.log2(i + 1)
        
        # Calculate IDCG (ideal DCG - all relevant items at top)
        ideal_relevance = sorted(relevance_list, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance, start=1):
            if rel > 0:
                idcg += rel / math.log2(i + 1)
        
        # Calculate NDCG
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return round(ndcg, 4)
    
    @staticmethod
    def evaluate(vectors_info: List[Dict], vectors_used: List[int]) -> Dict[str, Any]:
        """Run retrieval evaluation at Level 1 and Level 2."""
        # Level 1: Ranking Statistics (100%)
        hit_rate_3 = RetrievalEvaluator.calculate_hit_rate_at_k(vectors_info, vectors_used, k=3)
        hit_rate_5 = RetrievalEvaluator.calculate_hit_rate_at_k(vectors_info, vectors_used, k=5)
        hit_rate_10 = RetrievalEvaluator.calculate_hit_rate_at_k(vectors_info, vectors_used, k=10)
        mrr = RetrievalEvaluator.calculate_mrr(vectors_info, vectors_used)
        noise_ratio = RetrievalEvaluator.calculate_noise_ratio(vectors_info, vectors_used)
        
        # Level 2: Ranking Quality (100%)
        ndcg = RetrievalEvaluator.calculate_ndcg(vectors_info, vectors_used)
        
        return {
            'mrr': mrr,
            'ndcg': ndcg,
            'noise_ratio': noise_ratio,
            'hit_rate_at_3': hit_rate_3,
            'hit_rate_at_5': hit_rate_5,
            'hit_rate_at_10': hit_rate_10
        }


class EvaluationPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.level1_evaluator = Level1Evaluator()
        self.level2_evaluator = Level2Evaluator()
        self.level3_evaluator = Level3Evaluator(config)
        random.seed(config.random_seed)
    
    def should_run_level(self, level: int) -> bool:
        """Determine if a level should run based on sampling rate."""
        if level == 1:
            return random.random() < self.config.level1_sample_rate
        elif level == 2:
            return random.random() < self.config.level2_sample_rate
        elif level == 3:
            return random.random() < self.config.level3_sample_rate
        return False
    
    def estimate_cost(self, user_query: str, ai_response: str, context_chunks: List[str]) -> float:
        """Estimate cost based on token counts (rough estimate)."""
        # Rough estimate: $0.002 per 1K tokens for GPT-3.5
        total_tokens = len(user_query.split()) + len(ai_response.split())
        total_tokens += sum(len(chunk.split()) for chunk in context_chunks[:5])
        
        cost_per_1k = 0.002
        return round((total_tokens / 1000) * cost_per_1k, 6)
    
    def evaluate_single(self, user_query: str, ai_response: str, context_chunks: List[str], 
                       message_id: Optional[str] = None, vectors_info: Optional[List[Dict]] = None,
                       vectors_used: Optional[List[int]] = None, user_turn: Optional[Dict] = None,
                       ai_turn: Optional[Dict] = None, vectors_data: Optional[List[Dict]] = None) -> EvaluationResult:
        """Evaluate a single query-response pair."""
        start_time = time.time()
        
        metrics = {}
        
        # Level 1: Always run (or based on sample rate)
        if self.should_run_level(1):
            level1_metrics = self.level1_evaluator.evaluate(user_query, ai_response, context_chunks)
            metrics.update(level1_metrics)
        else:
            metrics['level1'] = None
        
        # Level 2: Sample-based
        if self.should_run_level(2):
            level2_metrics = self.level2_evaluator.evaluate(user_query, ai_response, context_chunks)
            if level2_metrics:
                metrics.update(level2_metrics)
            else:
                metrics['level2'] = None
        else:
            metrics['level2'] = None
        
        # Level 3: Tiny sample
        if self.should_run_level(3):
            level3_metrics = self.level3_evaluator.evaluate(user_query, ai_response, context_chunks)
            if level3_metrics:
                metrics.update(level3_metrics)
            else:
                metrics['level3'] = None
        else:
            metrics['level3'] = None
        
        # Retrieval Evaluation (always run if data available)
        retrieval_metrics = None
        if vectors_info is not None and vectors_used is not None:
            retrieval_metrics = RetrievalEvaluator.evaluate(vectors_info, vectors_used)
        
        # Operational Metrics (latency, cost, retrieval efficiency)
        operational_metrics = None
        if user_turn is not None and ai_turn is not None and vectors_data is not None and vectors_used is not None:
            operational_metrics = OperationalMetricsCalculator.calculate_operational_metrics(
                user_query, ai_response, user_turn, ai_turn, vectors_data, vectors_used
            )
        
        # Legacy metrics (for backward compatibility)
        latency_ms = operational_metrics.get('latency_ms') if operational_metrics else None
        cost_estimate = operational_metrics.get('cost_usd') if operational_metrics else None
        
        return EvaluationResult(
            message_id=message_id,
            user_query=user_query,
            ai_response=ai_response,
            metrics=metrics,
            retrieval_metrics=retrieval_metrics,
            operational_metrics=operational_metrics,
            latency_ms=latency_ms,
            cost_estimate=cost_estimate
        )
    
    def run_pipeline(self, conversation_path: str, context_vectors_path: str) -> List[Dict]:
        """Run the full evaluation pipeline."""
        # Map data
        mapped_data = DataMapper.map_data(conversation_path, context_vectors_path)
        
        if not mapped_data:
            print("Warning: No data could be mapped. Check file paths and structure.")
            return []
        
        results = []
        for data in mapped_data:
            result = self.evaluate_single(
                user_query=data['user_query'],
                ai_response=data['ai_response'],
                context_chunks=data['context_chunks'],
                message_id=data.get('message_id'),
                vectors_info=data.get('vectors_info'),
                vectors_used=data.get('vectors_used'),
                user_turn=data.get('user_turn'),
                ai_turn=data.get('ai_turn'),
                vectors_data=data.get('vectors_data')
            )
            results.append(asdict(result))
        
        return results


def create_dummy_data():
    """Create dummy JSON data for testing."""
    dummy_conversation = {
        "chat_id": 12345,
        "user_id": 67890,
        "conversation_turns": [
            {
                "turn": 1,
                "sender_id": 67890,
                "role": "User",
                "message": "What is the cost of IVF treatment?",
                "created_at": "2025-01-01T10:00:00.000000Z"
            },
            {
                "turn": 2,
                "sender_id": 1,
                "role": "AI/Chatbot",
                "message": "A complete IVF cycle at our clinic costs approximately Rs 3,00,000. This includes all medical procedures. Medications would typically cost about Rs 1,45,000 more.",
                "created_at": "2025-01-01T10:00:15.000000Z"
            }
        ]
    }
    
    dummy_context_vectors = {
        "status": "success",
        "status_code": 200,
        "data": {
            "vector_data": [
                {
                    "id": 1,
                    "source_url": "https://example.com/ivf-costs",
                    "text": "IVF treatment costs approximately Rs 3,00,000 for a complete cycle. This includes all medical procedures, egg retrieval, and embryo transfer. Additional medications cost around Rs 1,45,000.",
                    "tokens": 50,
                    "created_at": "2024-01-01T00:00:00.000Z"
                },
                {
                    "id": 2,
                    "source_url": "https://example.com/ivf-info",
                    "text": "The clinic offers comprehensive IVF services including ICSI and blastocyst transfer at no extra cost.",
                    "tokens": 30,
                    "created_at": "2024-01-01T00:00:00.000Z"
                }
            ],
            "sources": {
                "message_id": "msg_123",
                "vector_ids": [1, 2],
                "final_response": [
                    "A complete IVF cycle at our clinic costs approximately Rs 3,00,000. This includes all medical procedures. Medications would typically cost about Rs 1,45,000 more."
                ]
            }
        }
    }
    
    return dummy_conversation, dummy_context_vectors


def main():
    """Main execution block."""
    print("=" * 60)
    print("LLM Evaluation Pipeline - RAG System Assessment")
    print("=" * 60)
    
    # Initialize NLTK data if available
    if NLTK_AVAILABLE:
        try:
            # Download punkt_tab for newer NLTK versions (3.8+)
            try:
                nltk.download('punkt_tab', quiet=True)
            except:
                # Fallback to punkt for older versions
                try:
                    nltk.download('punkt', quiet=True)
                except:
                    pass
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
            print("Some features may not work correctly. Continuing anyway...")
    
    # Create configuration
    # Get OpenAI API key from environment variable or set it here
    import os
    openai_api_key = os.getenv('OPENAI_API_KEY')
    # run all the metrics on all the data with sample rate 1.0
    config = EvaluationConfig(
        level1_sample_rate=1.0,  # 100%
        level2_sample_rate=1.0,  # 10%
        level3_sample_rate=1.0,  # 1%
        openai_api_key=openai_api_key,  # Set from environment or pass directly
        openai_model="gpt-3.5-turbo"  # or "gpt-4" for better results
    )
    
    print(f"\nConfiguration:")
    print(f"  Level 1 (Simple) Sample Rate: {config.level1_sample_rate * 100}%")
    print(f"  Level 2 (Semantic) Sample Rate: {config.level2_sample_rate * 100}%")
    print(f"  Level 3 (LLM-based) Sample Rate: {config.level3_sample_rate * 100}%")
    if config.openai_api_key:
        print(f"  OpenAI API Key: {'*' * 20} (set)")
    else:
        print(f"  OpenAI API Key: Not set (Level 3 evaluation will not work)")
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(config)
    
    # Check if sample files exist, otherwise use dummy data
    import os
    conversation_path = "sample-chat-conversation-01.json"
    context_vectors_path = "sample_context_vectors-01.json"
    
    if os.path.exists(conversation_path) and os.path.exists(context_vectors_path):
        print(f"\nUsing sample files:")
        print(f"  Conversation: {conversation_path}")
        print(f"  Context Vectors: {context_vectors_path}")
    else:
        print("\nSample files not found. Creating dummy data for demonstration...")
        dummy_conv, dummy_ctx = create_dummy_data()
        
        # Save dummy data
        with open('dummy_conversation.json', 'w', encoding='utf-8') as f:
            json.dump(dummy_conv, f, indent=2)
        with open('dummy_context_vectors.json', 'w', encoding='utf-8') as f:
            json.dump(dummy_ctx, f, indent=2)
        
        conversation_path = 'dummy_conversation.json'
        context_vectors_path = 'dummy_context_vectors.json'
    
    # Run pipeline
    print("\nRunning evaluation pipeline...")
    results = pipeline.run_pipeline(conversation_path, context_vectors_path)
    
    # Output results
    print(f"\nEvaluated {len(results)} query-response pair(s).")
    print("\nResults:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Save results
    output_file = 'evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

