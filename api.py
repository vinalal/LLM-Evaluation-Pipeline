"""
FastAPI Backend for LLM Evaluation Pipeline
Handles synchronous fast metrics and asynchronous slow metrics
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Import the evaluation pipeline components
from llm_evaluation_pipeline import (
    EvaluationConfig,
    EvaluationPipeline,
    DataMapper,
    RetrievalEvaluator,
    OperationalMetricsCalculator
)

app = FastAPI(title="LLM Evaluation API", version="1.0.0")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task store (in production, use Redis)
task_store: Dict[str, Dict[str, Any]] = {}


class EvaluationRequest(BaseModel):
    """Request model for evaluation endpoint."""
    conversation_json: Dict[str, Any]
    context_vectors_json: Dict[str, Any]


class EvaluationResponse(BaseModel):
    """Response model for evaluation endpoint."""
    task_id: str
    status: str  # "partial" or "completed"
    immediate_results: List[Dict[str, Any]]


class TaskStatusResponse(BaseModel):
    """Response model for task status endpoint."""
    task_id: str
    status: str  # "processing" or "completed"
    level2_results: Optional[List[Dict[str, Any]]] = None
    level3_results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


def run_background_evaluation(
    task_id: str,
    mapped_data: List[Dict[str, Any]],
    config: EvaluationConfig
):
    """
    Background task to run Level 2 and Level 3 evaluations.
    Updates task_store when complete.
    """
    try:
        pipeline = EvaluationPipeline(config)
        
        level2_results = []
        level3_results = []
        
        for data in mapped_data:
            user_query = data['user_query']
            ai_response = data['ai_response']
            context_chunks = data['context_chunks']
            message_id = data.get('message_id')
            
            # Level 2: Medium (Semantic) - Run on sampled data
            if pipeline.should_run_level(2):
                level2_metrics = pipeline.level2_evaluator.evaluate(
                    user_query, ai_response, context_chunks
                )
                if level2_metrics:
                    level2_results.append({
                        'message_id': message_id,
                        'metrics': {'level2': level2_metrics}
                    })
            
            # Level 3: Robust (LLM-based) - Run on tiny sample
            if pipeline.should_run_level(3):
                level3_metrics = pipeline.level3_evaluator.evaluate(
                    user_query, ai_response, context_chunks
                )
                if level3_metrics:
                    level3_results.append({
                        'message_id': message_id,
                        'metrics': {'level3': level3_metrics}
                    })
        
        # Update task store with results
        task_store[task_id].update({
            'status': 'completed',
            'level2_results': level2_results,
            'level3_results': level3_results
        })
        
    except Exception as e:
        # Update task store with error
        task_store[task_id].update({
            'status': 'error',
            'error': str(e)
        })


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """
    Evaluate conversation and context vectors.
    Returns immediate Level 1 and Retrieval metrics.
    Offloads Level 2 and 3 to background tasks.
    """
    
    try:
        # Generate task ID
        
        task_id = str(uuid.uuid4())
        
        # Initialize task in store
        task_store[task_id] = {
            'status': 'processing',
            'level2_results': None,
            'level3_results': None,
            'error': None
        }
        
        # Map the data directly from JSON objects
        conversation = request.conversation_json
        context_vectors = request.context_vectors_json
        
        # Use DataMapper.map_data logic but with JSON objects instead of file paths
        # We'll manually extract the data since map_data expects file paths
        mapped_data = []
        # Extract final_response from context_vectors
        final_response = context_vectors.get('data', {}).get('final_response', '')
        print(final_response)
        
        if final_response:
            # Extract context chunks
            context_chunks = DataMapper.extract_context_chunks(context_vectors)
            message_id = context_vectors.get('data', {}).get('sources', {}).get('message_id')
            
            # Find matching turn
            matching_turn = DataMapper.find_matching_turn(conversation, final_response)
            
            if matching_turn:
                turn_number, turn_dict = matching_turn
                user_query = DataMapper.get_user_query_for_turn(conversation, turn_number)
                
                if user_query:
                    # Get turn dictionaries for operational metrics
                    user_turn = DataMapper.get_user_turn(conversation, turn_number)
                    ai_turn = DataMapper.get_ai_turn(conversation, turn_number)
                    
                    mapped_data.append({
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
                    })
            
            # Fallback: Use last user query
            if not mapped_data:
                last_user_query = DataMapper.get_last_user_query(conversation)
                last_user_turn = DataMapper.get_last_user_turn(conversation)
                
                if last_user_query:
                    # Try to find the AI turn that matches final_response
                    ai_turn = None
                    turns = conversation.get('conversation_turns', [])
                    for turn in reversed(turns):
                        if turn.get('role') in ['AI', 'AI/Chatbot', 'Chatbot']:
                            # Check if this turn's message matches final_response
                            if DataMapper._texts_match(final_response, turn.get('message', '')):
                                ai_turn = turn
                                break
                    
                    mapped_data.append({
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
                    })
        
        if not mapped_data:
            raise HTTPException(
                status_code=400,
                detail="Could not map conversation to context vectors. Check data structure."
            )
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Create configuration
        config = EvaluationConfig(
            level1_sample_rate=1.0,  # 100% - Always run
            level2_sample_rate=0.10,  # 10% - Sample
            level3_sample_rate=0.01,  # 1% - Tiny sample
            openai_api_key=openai_api_key,
            openai_model="gpt-3.5-turbo"
        )
        
        pipeline = EvaluationPipeline(config)
        
        # Run Level 1 and Retrieval metrics synchronously (fast)
        immediate_results = []
        
        for data in mapped_data:
            user_query = data['user_query']
            ai_response = data['ai_response']
            context_chunks = data['context_chunks']
            message_id = data.get('message_id')
            vectors_info = data.get('vectors_info')
            vectors_used = data.get('vectors_used')
            user_turn = data.get('user_turn')
            ai_turn = data.get('ai_turn')
            vectors_data = data.get('vectors_data', [])
            
            # Level 1: Simple metrics (fast)
            level1_metrics = pipeline.level1_evaluator.evaluate(
                user_query, ai_response, context_chunks
            )
            
            # Retrieval metrics (fast)
            retrieval_metrics = None
            if vectors_info is not None and vectors_used is not None:
                retrieval_metrics = RetrievalEvaluator.evaluate(vectors_info, vectors_used)
            
            # Operational metrics (latency, cost, retrieval efficiency)
            operational_metrics = None
            if user_turn is not None and ai_turn is not None and vectors_data and vectors_used is not None:
                operational_metrics = OperationalMetricsCalculator.calculate_operational_metrics(
                    user_query, ai_response, user_turn, ai_turn, vectors_data, vectors_used
                )
            
            immediate_results.append({
                'message_id': message_id,
                'metrics': {
                    'level1': level1_metrics,
                    'retrieval': retrieval_metrics
                },
                'operational_metrics': operational_metrics
            })
        
        # Offload Level 2 and 3 to background tasks
        background_tasks.add_task(
            run_background_evaluation,
            task_id,
            mapped_data,
            config
        )
        
        return EvaluationResponse(
            task_id=task_id,
            status="partial",
            immediate_results=immediate_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{task_id}", response_model=TaskStatusResponse)
async def get_results(task_id: str):
    """
    Get the status and results of a background evaluation task.
    """
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = task_store[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_data['status'],
        level2_results=task_data.get('level2_results'),
        level3_results=task_data.get('level3_results'),
        error=task_data.get('error')
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

