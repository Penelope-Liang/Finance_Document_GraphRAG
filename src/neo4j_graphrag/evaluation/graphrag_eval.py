"""
GraphRAG Evaluation Framework
Real-time recording of GraphRAG Q&A results, supports multiple datasets
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============== Live Results Writer ==============

class LiveResultsWriter:
    """Write evaluation results in real-time"""
    
    def __init__(self, output_dir: str, dataset_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = dataset_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Output files
        self.results_file = self.output_dir / f"live_results_{dataset_name}_{timestamp}.jsonl"
        self.summary_file = self.output_dir / f"summary_{dataset_name}_{timestamp}.json"
        self.log_file = self.output_dir / f"eval_{dataset_name}_{timestamp}.log"
        
        # Statistics
        self.total = 0
        self.start_time = time.time()
        
    def write_result(self, result: Dict[str, Any]):
        """Write a single result (append in real-time)"""
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def log(self, message: str):
        """Write to log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')
    
    def update_stats(self):
        """Update statistics"""
        self.total += 1
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        elapsed = time.time() - self.start_time
        
        return {
            "total_processed": self.total,
            "elapsed_seconds": elapsed,
            "questions_per_second": self.total / elapsed if elapsed > 0 else 0,
        }
    
    def write_summary(self):
        """Write final summary"""
        stats = self.get_current_stats()
        stats["dataset"] = self.dataset_name
        stats["timestamp"] = datetime.now().isoformat()
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return stats


# ============== Evaluator ==============

class GraphRAGEvaluator:
    """GraphRAG Evaluator"""
    
    def __init__(
        self,
        dataset_name: str,
        output_dir: str = "evaluation/results",
    ):
        self.dataset_name = dataset_name
        self.writer = LiveResultsWriter(output_dir, dataset_name)
        
    def load_questions(self, questions_file: str) -> List[Dict]:
        """Load questions data"""
        path = Path(questions_file)
        if path.suffix.lower() == ".jsonl":
            rows: List[Dict[str, Any]] = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    rows.append(json.loads(raw))
            return rows

        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("questions", "data", "items", "rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported question file format: {questions_file}")
    
    def record_single(
        self,
        question_data: Dict,
        predicted_answer: str,
        retrieved_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record result for a single question"""
        question = question_data.get("question") or question_data.get("text", "")
        gold_answer = question_data.get("gold_answer", question_data.get("answer", ""))
        question_id = question_data.get("id") or question_data.get("_id", "")
        
        # Update stats
        self.writer.update_stats()
        
        # Build result
        result = {
            "id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "progress": self.writer.total,
        }
        
        if retrieved_context:
            result["context_preview"] = retrieved_context[:500] + "..." if len(retrieved_context) > 500 else retrieved_context
        
        # Write result in real-time
        self.writer.write_result(result)
        
        # Print real-time status
        self.writer.log(f"[{self.writer.total}] Q: {question[:50]}...")
        
        return result
    
    def finalize(self) -> Dict[str, Any]:
        """Finish evaluation, output final summary"""
        stats = self.writer.write_summary()
        
        self.writer.log("=" * 60)
        self.writer.log("EVALUATION COMPLETE")
        self.writer.log(f"Dataset: {self.dataset_name}")
        self.writer.log(f"Total Questions: {stats['total_processed']}")
        self.writer.log(f"Time Elapsed: {stats['elapsed_seconds']:.1f}s")
        self.writer.log("=" * 60)
        
        return stats


# ============== Example Usage ==============

def example_usage():
    """Example: How to use the evaluation framework"""
    
    # 1. Initialize evaluator
    evaluator = GraphRAGEvaluator(
        dataset_name="medical",
        output_dir="evaluation/results",
    )
    
    # 2. Load questions
    questions = evaluator.load_questions("dataset/medical/medical_questions.json")
    
    # 3. Record each question
    for i, q in enumerate(questions):
        # TODO: Call your GraphRAG system to get the answer
        # predicted_answer = graphrag.search(q["question"])
        predicted_answer = "placeholder answer"  # Placeholder
        
        # Record result
        evaluator.record_single(
            question_data=q,
            predicted_answer=predicted_answer,
            retrieved_context="Retrieved context...",  # Optional
        )
    
    # 4. Finish
    final_stats = evaluator.finalize()
    print(f"\nTotal: {final_stats['total_processed']}")


if __name__ == "__main__":
    example_usage()
