#!/usr/bin/env python3
"""
Batch Inference Runner for SpatioTemporal Data Generation

Generates many independent scenarios concurrently via an OpenAI-compatible LLM
API (configured through ``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL``).

Strategy:
1. Pre-generate all task configurations (domain + num_nodes combinations)
2. For each task, execute agents in batches:
   - Batch 1: All Agent 1 (scenario generation) calls
   - Process Judge 1 results, identify which need regeneration
   - Batch 2: Agent 2 (parsing) for approved scenarios + Agent 1 retries
   - Continue iteratively until all tasks complete
3. Track task state and dynamically adjust batch sizes
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pickle
import numpy as np

# Import NetworkSDEGenerator from demo_sts_sde.py to reuse data generation logic
from demo_sts_sde import NetworkSDEGenerator

# Configuration
OUTPUT_DIR = "batch_output"
BATCH_INPUT_DIR = "batch_inputs"

# Import prompts
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import (
    SCENARIO_GENERATION_PROMPT,
    SCENARIO_PARSING_PROMPT,
    SDE_PARAMETERS_PROMPT,
    TIME_VARYING_ADJACENCY_PROMPT,
    JUDGE_SCENARIO_PARSING_PROMPT,
    JUDGE_PARAMETER_VALIDATION_PROMPT
)

from llm_client import LLMClient

# Shared LLM client (reads LLM_API_KEY / LLM_BASE_URL / LLM_MODEL from env).
llm_client = LLMClient()


class TaskState:
    """Represents the state of a single generation task with detailed progress tracking."""
    
    # Define the complete pipeline stages
    PIPELINE_STAGES = ["agent1", "agent2", "judge1", "agent3", "agent4", "judge2", "complete"]
    
    def __init__(self, task_id: str, num_nodes: int, domain: str):
        self.task_id = task_id
        self.num_nodes = num_nodes
        self.domain = domain
        
        # Agent outputs
        self.agent1_scenario = None
        self.agent1_iteration = 0
        self.agent2_structured_json = None
        self.agent2_iteration = 0
        self.judge1_approved = False
        self.judge1_feedback = None
        
        self.agent3_sde_params = None
        self.agent4_adjacency = None
        self.agent34_iteration = 0
        self.judge2_approved = False
        self.judge2_feedback = None
        
        # Enhanced state tracking
        self.current_stage = "agent1"  # Current stage in pipeline
        self.next_stage = "agent1"     # Next stage to process
        self.is_ready = True           # Ready to be processed
        self.is_processing = False     # Currently being processed in a batch
        self.completed = False
        self.failed = False
        self.error_message = None
        
        # Progress tracking
        self.stage_history = []        # List of (stage, timestamp, status) tuples
        self.total_api_calls = 0       # Total API calls made for this task
        self.created_at = datetime.now()
        self.completed_at = None
        
        # Record initial state
        self._record_stage_change("agent1", "initialized")
    
    def _record_stage_change(self, stage: str, status: str):
        """Record a stage change for progress tracking."""
        self.stage_history.append({
            "stage": stage,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "iteration": self._get_iteration_for_stage(stage)
        })
    
    def _get_iteration_for_stage(self, stage: str) -> int:
        """Get current iteration number for a stage."""
        if stage in ["agent1", "agent2", "judge1"]:
            return max(self.agent1_iteration, self.agent2_iteration)
        elif stage in ["agent3", "agent4", "judge2"]:
            return self.agent34_iteration
        return 0
    
    def set_next_stage(self, stage: str, feedback: str = None):
        """Set the next stage to process and mark as ready."""
        self.next_stage = stage
        self.is_ready = True
        self.is_processing = False
        self._record_stage_change(stage, "queued")
        
        if feedback:
            if stage in ["agent1", "agent2"]:
                self.judge1_feedback = feedback
            elif stage in ["agent3", "agent4"]:
                self.judge2_feedback = feedback
    
    def mark_processing(self):
        """Mark task as currently being processed."""
        self.is_ready = False
        self.is_processing = True
        self._record_stage_change(self.next_stage, "processing")
    
    def mark_complete(self):
        """Mark task as fully completed."""
        self.completed = True
        self.is_ready = False
        self.is_processing = False
        self.next_stage = "complete"
        self.current_stage = "complete"
        self.completed_at = datetime.now()
        self._record_stage_change("complete", "success")
    
    def mark_failed(self, error: str):
        """Mark task as failed."""
        self.failed = True
        self.is_ready = False
        self.is_processing = False
        self.error_message = error
        self.completed_at = datetime.now()
        self._record_stage_change(self.current_stage, f"failed: {error}")
    
    def get_progress_percentage(self) -> float:
        """Calculate completion percentage based on pipeline stages."""
        if self.failed:
            return 0.0
        if self.completed:
            return 100.0
        
        # Calculate based on current stage position in pipeline
        try:
            stage_idx = self.PIPELINE_STAGES.index(self.current_stage)
            return (stage_idx / len(self.PIPELINE_STAGES)) * 100
        except ValueError:
            return 0.0
    
    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary."""
        if self.failed:
            return f"❌ Failed at {self.current_stage}: {self.error_message}"
        if self.completed:
            duration = (self.completed_at - self.created_at).total_seconds()
            return f"✅ Complete ({duration:.1f}s, {self.total_api_calls} API calls)"
        
        stage_display = {
            "agent1": "Generating scenario",
            "agent2": "Parsing scenario",
            "judge1": "Validating scenario",
            "agent3": "Generating SDE params",
            "agent4": "Generating adjacency",
            "judge2": "Validating parameters"
        }
        
        status = "🔄 Processing" if self.is_processing else "⏳ Queued"
        stage_name = stage_display.get(self.next_stage, self.next_stage)
        progress = self.get_progress_percentage()
        
        return f"{status} [{progress:.0f}%] {stage_name} (iter: {self._get_iteration_for_stage(self.next_stage)})"
    
    def to_dict(self):
        """Serialize task state to dictionary."""
        return {
            "task_id": self.task_id,
            "num_nodes": self.num_nodes,
            "domain": self.domain,
            "current_stage": self.current_stage,
            "next_stage": self.next_stage,
            "is_ready": self.is_ready,
            "is_processing": self.is_processing,
            "completed": self.completed,
            "failed": self.failed,
            "progress_percentage": self.get_progress_percentage(),
            "progress_summary": self.get_progress_summary(),
            "agent1_iteration": self.agent1_iteration,
            "agent2_iteration": self.agent2_iteration,
            "agent34_iteration": self.agent34_iteration,
            "total_api_calls": self.total_api_calls,
            "judge1_approved": self.judge1_approved,
            "judge2_approved": self.judge2_approved,
            "stage_history": self.stage_history
        }


class ProgressMonitor:
    """Real-time progress monitoring for all tasks."""
    
    def __init__(self):
        self.last_print_time = datetime.now()
        self.print_interval = 10  # Print summary every 10 seconds
    
    def print_progress(self, tasks: Dict[str, TaskState], force: bool = False):
        """Print progress summary for all tasks."""
        now = datetime.now()
        if not force and (now - self.last_print_time).total_seconds() < self.print_interval:
            return
        
        self.last_print_time = now
        
        # Calculate statistics
        total = len(tasks)
        completed = sum(1 for t in tasks.values() if t.completed)
        failed = sum(1 for t in tasks.values() if t.failed)
        processing = sum(1 for t in tasks.values() if t.is_processing)
        queued = sum(1 for t in tasks.values() if t.is_ready and not t.is_processing)
        
        # Count by stage
        stage_counts = {}
        for task in tasks.values():
            if not task.completed and not task.failed:
                stage = task.next_stage
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        # Average progress
        avg_progress = sum(t.get_progress_percentage() for t in tasks.values()) / total if total > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"PROGRESS SUMMARY [{now.strftime('%H:%M:%S')}]")
        print(f"{'='*80}")
        print(f"Overall: {avg_progress:.1f}% | ✅ {completed}/{total} | ❌ {failed} | 🔄 {processing} | ⏳ {queued}")
        
        if stage_counts:
            print(f"\nBy Stage:")
            for stage, count in sorted(stage_counts.items()):
                print(f"  {stage}: {count} tasks")
        
        # Show sample tasks
        print(f"\nRecent Activity (last 5 tasks):")
        recent_tasks = sorted(
            [t for t in tasks.values() if not t.completed],
            key=lambda t: len(t.stage_history),
            reverse=True
        )[:5]
        
        for task in recent_tasks:
            print(f"  {task.task_id}: {task.get_progress_summary()}")
        
        print(f"{'='*80}\n")


class BatchInferenceRunner:
    """Queue-driven batch inference runner with unified job submission and hybrid execution model."""
    
    def __init__(self, domains: List[str], node_counts: List[int], min_tasks: int = 100,
                 judge1_max_outer_iterations: int = 3,
                 judge1_max_inner_iterations: int = 2,
                 judge2_max_iterations: int = 5,
                 switch_to_realtime_threshold: int = 10):
        self.domains = domains
        self.node_counts = node_counts
        self.min_tasks = min_tasks
        self.tasks: Dict[str, TaskState] = {}
        
        # Judge Agent iteration limits (matching demo_sts_sde.py configuration)
        self.judge1_max_outer_iterations = judge1_max_outer_iterations  # Agent 1 scenario regeneration
        self.judge1_max_inner_iterations = judge1_max_inner_iterations  # Agent 2 parsing correction
        self.judge2_max_iterations = judge2_max_iterations              # Agent 3&4 parameter regeneration
        
        # Hybrid execution model configuration
        self.switch_to_realtime_threshold = switch_to_realtime_threshold
        self.llm = llm_client

        # Rate limiting configuration for real-time API
        self.realtime_request_interval = 0.0  # Seconds between requests (rely on LLMClient retries)
        self.last_realtime_request_time = 0

        print(f"\n{'='*80}")
        print(f"HYBRID EXECUTION MODEL CONFIGURATION")
        print(f"{'='*80}")
        print(f"Batch Mode: >={self.switch_to_realtime_threshold} tasks pending (concurrent LLM calls)")
        print(f"Real-time Mode: <{self.switch_to_realtime_threshold} tasks pending (sequential LLM calls)")
        print(f"\nJudge Agent Configuration:")
        print(f"  - Judge 1 Max Outer: {self.judge1_max_outer_iterations}")
        print(f"  - Judge 1 Max Inner: {self.judge1_max_inner_iterations}")
        print(f"  - Judge 2 Max: {self.judge2_max_iterations}")
        print(f"{'='*80}\n")
        
        # Progress monitoring
        self.progress_monitor = ProgressMonitor()
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(BATCH_INPUT_DIR, exist_ok=True)
        
        # Initialize tasks
        self._initialize_tasks()
        
    def _initialize_tasks(self):
        """
        Create independent task states to reach min_tasks.
        Each task is an independent sample that needs its own unique data generation,
        even if domain×node combinations are repeated.
        """
        combinations = [(num_nodes, domain) for num_nodes in self.node_counts for domain in self.domains]
        
        if not combinations:
            raise ValueError("No domain/node_count combinations provided!")
        
        # Create exactly min_tasks independent tasks
        # Cycle through combinations to ensure distribution
        for task_id in range(self.min_tasks):
            combo_idx = task_id % len(combinations)
            num_nodes, domain = combinations[combo_idx]
            
            # Each task gets a unique ID - these are all REAL independent samples
            task_key = f"task_{task_id:04d}"
            self.tasks[task_key] = TaskState(task_key, num_nodes, domain)
        
        print(f"Initialized {len(self.tasks)} independent tasks")
        print(f"  - Node counts: {self.node_counts}")
        print(f"  - Domains: {self.domains}")
        print(f"  - Unique combinations: {len(combinations)}")
        print(f"  - Each task is an independent sample requiring unique data generation")
        
        # Show distribution
        combo_counts = {}
        for task in self.tasks.values():
            key = f"{task.domain}_{task.num_nodes}nodes"
            combo_counts[key] = combo_counts.get(key, 0) + 1
        
        print(f"\n  Task distribution across combinations:")
        for combo, count in sorted(combo_counts.items()):
            print(f"    {combo}: {count} independent samples")
    
    def get_ready_tasks(self, limit: int = 100) -> List[Tuple[str, TaskState]]:
        """
        Get up to 'limit' tasks that are ready to be processed.
        Returns list of (task_key, task) tuples.
        """
        ready_tasks = []
        for task_key, task in self.tasks.items():
            if task.is_ready and not task.is_processing and not task.completed and not task.failed:
                ready_tasks.append((task_key, task))
                if len(ready_tasks) >= limit:
                    break
        
        return ready_tasks
    
    def create_mixed_batch_requests(self, tasks: List[Tuple[str, TaskState]]) -> List[Dict]:
        """
        Create batch requests for a mixed set of tasks at different stages.
        This is the core of the queue-driven approach - tasks can be at any stage.
        """
        requests = []
        
        for task_key, task in tasks:
            stage = task.next_stage
            
            try:
                if stage == "agent1":
                    prompt = self._create_agent1_prompt(task)
                    request = self._create_bedrock_request(
                        record_id=f"{task_key}_agent1_iter{task.agent1_iteration}",
                        prompt=prompt,
                        max_tokens=2048
                    )
                    requests.append(request)
                    task.mark_processing()
                
                elif stage == "agent2":
                    prompt = self._create_agent2_prompt(task)
                    request = self._create_bedrock_request(
                        record_id=f"{task_key}_agent2_iter{task.agent2_iteration}",
                        prompt=prompt,
                        max_tokens=4096
                    )
                    requests.append(request)
                    task.mark_processing()
                
                elif stage == "judge1":
                    prompt = self._create_judge1_prompt(task)
                    request = self._create_bedrock_request(
                        record_id=f"{task_key}_judge1_iter{task.agent2_iteration}",
                        prompt=prompt,
                        max_tokens=2000
                    )
                    requests.append(request)
                    task.mark_processing()
                
                elif stage == "agent3":
                    prompt = self._create_agent3_prompt(task)
                    request = self._create_bedrock_request(
                        record_id=f"{task_key}_agent3_iter{task.agent34_iteration}",
                        prompt=prompt,
                        max_tokens=2000
                    )
                    requests.append(request)
                    task.mark_processing()
                
                elif stage == "agent4":
                    prompt = self._create_agent4_prompt(task)
                    request = self._create_bedrock_request(
                        record_id=f"{task_key}_agent4_iter{task.agent34_iteration}",
                        prompt=prompt,
                        max_tokens=2000
                    )
                    requests.append(request)
                    task.mark_processing()
                
                elif stage == "judge2":
                    viz_base64 = self._generate_preview_visualization(task)
                    if viz_base64:
                        prompt = self._create_judge2_prompt(task)
                        request = self._create_bedrock_multimodal_request(
                            record_id=f"{task_key}_judge2_iter{task.agent34_iteration}",
                            prompt=prompt,
                            image_base64=viz_base64,
                            max_tokens=2000
                        )
                        requests.append(request)
                        task.mark_processing()
                    else:
                        print(f"Warning: Failed to generate visualization for {task_key}, skipping")
                
            except Exception as e:
                print(f"Error creating request for {task_key} at stage {stage}: {e}")
                task.mark_failed(str(e))
        
        return requests
    
    def create_batch_requests(self, stage: str) -> List[Dict]:
        """
        Create batch requests for a specific stage.
        
        Args:
            stage: The agent stage to create requests for
        
        Returns:
            List of batch request dictionaries
        """
        requests = []
        
        if stage == "agent1":
            # Agent 1: Scenario Generation
            for task_key, task in self.tasks.items():
                if task.current_stage == "agent1" and not task.completed and not task.failed:
                    prompt = self._create_agent1_prompt(task)
                    request = self._create_bedrock_request(
                        record_id=f"{task_key}_agent1_iter{task.agent1_iteration}",
                        prompt=prompt,
                        max_tokens=2048
                    )
                    requests.append(request)
        
        elif stage == "agent2":
            # Agent 2: Scenario Parsing
            for task_key, task in self.tasks.items():
                if task.current_stage == "agent2" and not task.completed and not task.failed:
                    prompt = self._create_agent2_prompt(task)
                    request = self._create_bedrock_request(
                        record_id=f"{task_key}_agent2_iter{task.agent2_iteration}",
                        prompt=prompt,
                        max_tokens=4096
                    )
                    requests.append(request)
        
        elif stage == "agent3":
            # Agent 3: SDE Parameters
            for task_key, task in self.tasks.items():
                if task.current_stage == "agent3" and not task.completed and not task.failed:
                    prompt = self._create_agent3_prompt(task)
                    request = self._create_bedrock_request(
                        record_id=f"{task_key}_agent3_iter{task.agent34_iteration}",
                        prompt=prompt,
                        max_tokens=2000  #ok
                    )
                    requests.append(request)
        
        elif stage == "agent4":
            # Agent 4: Time-varying Adjacency
            for task_key, task in self.tasks.items():
                if task.current_stage == "agent4" and not task.completed and not task.failed:
                    prompt = self._create_agent4_prompt(task)
                    request = self._create_bedrock_request(
                        record_id=f"{task_key}_agent4_iter{task.agent34_iteration}",
                        prompt=prompt,
                        max_tokens=2000
                    )
                    requests.append(request)
        
        elif stage == "judge1":
            # Judge Agent 1: Validate scenario parsing
            for task_key, task in self.tasks.items():
                if task.current_stage == "judge1" and not task.completed and not task.failed:
                    prompt = self._create_judge1_prompt(task)
                    request = self._create_bedrock_request(
                        record_id=f"{task_key}_judge1_iter{task.agent2_iteration}",
                        prompt=prompt,
                        max_tokens=2000
                    )
                    requests.append(request)
        
        elif stage == "judge2":
            # Judge Agent 2: Validate parameters with visualization
            for task_key, task in self.tasks.items():
                if task.current_stage == "judge2" and not task.completed and not task.failed:
                    # Generate visualization and create multimodal request
                    viz_base64 = self._generate_preview_visualization(task)
                    if viz_base64:
                        prompt = self._create_judge2_prompt(task)
                        request = self._create_bedrock_multimodal_request(
                            record_id=f"{task_key}_judge2_iter{task.agent34_iteration}",
                            prompt=prompt,
                            image_base64=viz_base64,
                            max_tokens=2000
                        )
                        requests.append(request)
        
        return requests
    
    def _create_agent1_prompt(self, task: TaskState) -> str:
        """Create Agent 1 (scenario generation) prompt."""
        domain_hints = {
            'Transportation': 'traffic flow, vehicle movement, or transportation networks',
            'Energy': 'power grid, energy consumption, or renewable energy generation',
            'Environment&Pollution': 'air quality, pollution levels, or environmental monitoring',
            'Ecology': 'ecosystem dynamics, species populations, or ecological networks',
            'Public Health': 'disease spread, infection rates, or public health surveillance',
            'Hydrology': 'water flow, river networks, or hydrological cycles',
            'Oceanography': 'ocean currents, marine ecosystems, or oceanographic data',
            'Agriculture': 'crop yields, agricultural production, or farming networks',
            'Mobility': 'human mobility, migration patterns, or movement networks',
            'Climate': 'weather patterns, climate data, or atmospheric conditions'
        }
        
        domain_hint = f"\n\nPreferred domain/context: {domain_hints.get(task.domain, task.domain)}"
        
        prompt = SCENARIO_GENERATION_PROMPT.format(
            num_nodes=task.num_nodes,
            max_seq_len=365
        ) + domain_hint
        
        # Add revision instructions if this is a retry
        if task.agent1_iteration > 0 and task.agent1_scenario and task.judge1_feedback:
            revision_section = f"\n\n{'='*60}\nSCENARIO REVISION MODE\n{'='*60}\n\nYour task is to REVISE the following scenario based on judge feedback.\n\n**PREVIOUS SCENARIO:**\n{task.agent1_scenario}\n\n{'='*60}\nJUDGE FEEDBACK:\n{'='*60}\n{task.judge1_feedback}\n{'='*60}\n\nPlease revise the above scenario to address the judge's feedback.\n"
            prompt = prompt + revision_section
        
        return prompt
    
    def _create_agent2_prompt(self, task: TaskState) -> str:
        """Create Agent 2 (scenario parsing) prompt."""
        prompt = SCENARIO_PARSING_PROMPT.replace("{scenario}", task.agent1_scenario)
        
        # Add feedback if this is a retry
        if task.agent2_iteration > 0 and task.judge1_feedback:
            feedback_section = f"\n\n{'='*60}\nPREVIOUS ATTEMPT FEEDBACK FROM JUDGE:\n{'='*60}\n{task.judge1_feedback}\n{'='*60}\n\nPlease address the issues mentioned above in your new parsing.\n"
            prompt = prompt + feedback_section
        
        return prompt
    
    def _create_agent3_prompt(self, task: TaskState) -> str:
        """Create Agent 3 (SDE parameters) prompt."""
        structured_json_str = json.dumps(task.agent2_structured_json, indent=2, ensure_ascii=False)
        prompt = SDE_PARAMETERS_PROMPT.format(structured_scenario=structured_json_str)
        
        # Add feedback if this is a retry
        if task.agent34_iteration > 0 and task.judge2_feedback:
            feedback_section = f"\n\n{'='*60}\nPREVIOUS ATTEMPT FEEDBACK FROM JUDGE:\n{'='*60}\n{task.judge2_feedback}\n{'='*60}\n\nPlease address the issues mentioned above.\n"
            prompt = prompt + feedback_section
        
        return prompt
    
    def _create_agent4_prompt(self, task: TaskState) -> str:
        """Create Agent 4 (time-varying adjacency) prompt."""
        structured_json_str = json.dumps(task.agent2_structured_json, indent=2, ensure_ascii=False)
        prompt = TIME_VARYING_ADJACENCY_PROMPT.format(structured_scenario=structured_json_str)
        
        return prompt
    
    def _create_judge1_prompt(self, task: TaskState) -> str:
        """Create Judge Agent 1 (scenario parsing validation) prompt."""
        parsed_json_str = json.dumps(task.agent2_structured_json, indent=2, ensure_ascii=False)
        
        prompt = JUDGE_SCENARIO_PARSING_PROMPT.format(
            expected_num_nodes=task.num_nodes,
            scenario=task.agent1_scenario,
            parsed_json=parsed_json_str
        )
        
        return prompt
    
    def _create_judge2_prompt(self, task: TaskState) -> str:
        """Create Judge Agent 2 (parameter validation) prompt."""
        structured_scenario_str = json.dumps(task.agent2_structured_json, indent=2, ensure_ascii=False)
        sde_params_str = json.dumps(task.agent3_sde_params, indent=2, ensure_ascii=False)
        adjacency_str = json.dumps(task.agent4_adjacency, indent=2, ensure_ascii=False)
        
        previous_assessment_section = ""
        if task.agent34_iteration > 1 and task.judge2_feedback:
            previous_assessment_section = f"""
**Previous Assessment (Iteration {task.agent34_iteration - 1}):**
Please review your previous feedback and check if the new parameters have addressed these issues.
```
{task.judge2_feedback}
```
"""
        
        prompt = JUDGE_PARAMETER_VALIDATION_PROMPT.format(
            structured_scenario=structured_scenario_str,
            sde_parameters=sde_params_str,
            time_varying_adjacency=adjacency_str,
            previous_assessment_section=previous_assessment_section
        )
        
        return prompt
    
    def _create_bedrock_request(self, record_id: str, prompt: str, max_tokens: int = 4096) -> Dict:
        """Create a single LLM request entry (kept legacy name for call sites)."""
        return {
            "recordId": record_id,
            "modelInput": {
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            },
        }

    def _create_bedrock_multimodal_request(self, record_id: str, prompt: str, image_base64: str, max_tokens: int = 4096) -> Dict:
        """Create a multimodal LLM request entry (text + one image)."""
        return {
            "recordId": record_id,
            "modelInput": {
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                        ],
                    }
                ],
            },
        }
    
    def _calculate_seq_len_from_scenario(self, structured_scenario: Dict) -> int:
        """
        Calculate sequence length from Agent 2's structured scenario.
        Prioritizes the 'seq_len' field if present, otherwise calculates from time_span and sampling_frequency.
        """
        import re
        
        # Priority 1: Use seq_len field directly (added in Agent 2 prompt update)
        if "seq_len" in structured_scenario:
            seq_len = structured_scenario["seq_len"]
            if isinstance(seq_len, (int, float)):
                return int(seq_len)
        
        # Priority 2: Try to get time_span and sampling_frequency
        time_span_str = structured_scenario.get("time_span", "")
        sampling_freq_str = structured_scenario.get("sampling_frequency", "1 day")
        
        if time_span_str:
            # Parse time_span (e.g., "1 year", "48 hours", "7 days", "3 months")
            time_value, time_unit = self._parse_time_string(time_span_str)
            sampling_value, sampling_unit = self._parse_time_string(sampling_freq_str)
            
            if time_value and time_unit and sampling_value and sampling_unit:
                # Convert to common unit (days) and calculate seq_len
                time_in_days = self._convert_to_days(time_value, time_unit)
                sampling_in_days = self._convert_to_days(sampling_value, sampling_unit)
                
                if sampling_in_days > 0:
                    seq_len = int(time_in_days / sampling_in_days)
                    return seq_len
        
        # Priority 3: Try other fields
        if "duration" in structured_scenario:
            duration = structured_scenario["duration"]
            if isinstance(duration, (int, float)):
                return int(duration)
        
        if "sequence_length" in structured_scenario:
            return int(structured_scenario["sequence_length"])
        
        # Priority 4: Try repeat_period as fallback
        if "drift_patterns" in structured_scenario:
            patterns = structured_scenario["drift_patterns"]
            if isinstance(patterns, dict) and "repeat_period" in patterns:
                return int(patterns["repeat_period"])
        
        # Default fallback
        print(f"Warning: Could not extract seq_len from scenario, using default 48")
        return 48
    
    def _parse_time_string(self, time_str: str) -> tuple:
        """Parse time string like '1 year' or '48 hours' into (value, unit)."""
        import re
        if not isinstance(time_str, str):
            return (None, None)
        
        # Extract number and unit
        match = re.search(r'(\d+\.?\d*)\s*(\w+)', time_str.lower())
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            return (value, unit)
        return (None, None)
    
    def _convert_to_days(self, value: float, unit: str) -> float:
        """Convert time value to days."""
        unit = unit.lower().rstrip('s')  # Remove plural 's'
        
        conversions = {
            'hour': 1/24,
            'day': 1,
            'week': 7,
            'month': 30,  # Approximate
            'year': 365
        }
        
        return value * conversions.get(unit, 1)
    
    def _generate_preview_visualization(self, task: TaskState) -> str:
        """Generate a preview visualization for Judge Agent 2 and return as base64."""
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        try:
            # Import necessary components from demo_sts_sde
            from demo_sts_sde import NetworkSDEGenerator
            
            # Extract sequence length from structured scenario
            seq_len = self._calculate_seq_len_from_scenario(task.agent2_structured_json)
            
            # Create a temporary generator instance (only used for SDE simulation, not LLM calls)
            temp_gen = NetworkSDEGenerator(num_nodes=task.num_nodes)
            temp_gen.seq_len = seq_len
            temp_gen.sde_params = task.agent3_sde_params
            
            # Enforce base_adjacency constraint (matching demo_sts_sde.py logic)
            # Create base adjacency matrix from structured_scenario edges
            base_adj = np.zeros((task.num_nodes, task.num_nodes))
            for edge in task.agent2_structured_json.get('edges', []):
                source = edge['source']
                target = edge['target']
                base_adj[source, target] = 0.1  # All edges = 0.1
            
            # Update agent4_adjacency with the enforced base_adjacency
            task.agent4_adjacency["base_adjacency"] = base_adj.tolist()
            
            # Extract edge lags from structured scenario
            edge_lags = {}
            for edge in task.agent2_structured_json.get('edges', []):
                if 'time_lag' in edge:
                    edge_key = f"{edge['source']}->{edge['target']}"
                    edge_lags[edge_key] = edge['time_lag']
            
            # Build network_sde structure
            network_sde = {
                "structured_scenario": task.agent2_structured_json,
                "sequence_length": seq_len,
                "sde_parameters": task.agent3_sde_params,
                "time_varying_adjacency": task.agent4_adjacency,
                "dt": 1.0,
                "noise_correlation": np.eye(task.num_nodes),
                "edge_lags": edge_lags
            }
            
            # Generate time series data
            # Note: generate_spatiotemporal_data will add adjacency_matrices to network_sde
            ts_data, generation_info = temp_gen.generate_spatiotemporal_data(network_sde)
            
            # Extract adjacency matrices from generation_info if available
            if "adjacency_matrices" in generation_info:
                network_sde["adjacency_matrices"] = generation_info["adjacency_matrices"]
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            for i in range(task.num_nodes):
                node_name = task.agent2_structured_json["nodes"][i]["name"]
                ax.plot(ts_data[i], label=f'Node {i}: {node_name[:20]}', linewidth=2)
            
            ax.set_title('Time Series Preview for Judge Validation', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to bytes and encode as base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return image_base64
            
        except Exception as e:
            print(f"Warning: Failed to generate visualization for {task.task_id}: {e}")
            return None
    
    def _pad_batch_to_minimum(self, requests: List[Dict], min_size: int = 100) -> List[Dict]:
        """No-op kept for backwards compatibility (no batch-API minimum any more)."""
        return requests

    
    def submit_batch_job(self, requests: List[Dict], stage_name: str) -> Tuple[str, str]:
        """
        Run all requests concurrently via the LLM API and persist a JSONL copy.

        Returns ``(local_jsonl_path, local_jsonl_path)`` so the existing call
        sites (which expect a job identifier and an output URI) keep working.
        """
        if not requests:
            print(f"No requests for stage {stage_name}, skipping...")
            return None, None

        print(f"\n{'='*80}")
        print(f"Running stage: {stage_name}")
        print(f"Total LLM requests: {len(requests)}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(BATCH_INPUT_DIR, exist_ok=True)
        batch_file = os.path.join(BATCH_INPUT_DIR, f"{stage_name}_{timestamp}.jsonl")
        with open(batch_file, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
        print(f"Saved request batch to {batch_file}")

        results = self.llm.run_batch(requests, max_workers=8)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_file = os.path.join(OUTPUT_DIR, f"{stage_name}_{timestamp}_results.jsonl")
        with open(results_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"✓ Stage {stage_name} produced {len(results)} results -> {results_file}")
        print(f"{'='*80}\n")

        # Cache results so download_results can return them by handle.
        self._stage_results_cache = getattr(self, "_stage_results_cache", {})
        self._stage_results_cache[results_file] = results

        return results_file, results_file

    def wait_for_job(self, job_arn: str, check_interval: int = 60) -> Dict:
        """No-op (kept for API compatibility – the work was already done)."""
        return {"status": "Completed"}

    def download_results(self, output_s3_uri: str) -> List[Dict]:
        """Return cached batch results (already produced by ``submit_batch_job``)."""
        cache = getattr(self, "_stage_results_cache", {})
        if output_s3_uri in cache:
            return cache.pop(output_s3_uri)
        if output_s3_uri and os.path.isfile(output_s3_uri):
            results = []
            with open(output_s3_uri, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return results
        return []

    
    def process_results(self, results: List[Dict], stage: str):
        """Process batch results and update task states."""
        print(f"\nProcessing {len(results)} results for stage: {stage}")
        
        # Filter out padding (handle missing recordId gracefully)
        real_results = []
        for r in results:
            record_id = r.get('recordId') or r.get('record_id', '')
            if '_pad' not in record_id:
                real_results.append(r)
        
        print(f"Real results (non-padded): {len(real_results)}")
        
        for result in real_results:
            record_id = result.get('recordId') or result.get('record_id', '')
            
            if not record_id:
                print(f"Warning: Result has no recordId, skipping")
                continue
            
            # Extract task_key
            task_key = record_id.split('_agent')[0].split('_judge')[0]
            
            if task_key not in self.tasks:
                print(f"Warning: Unknown task_key {task_key}")
                continue
            
            task = self.tasks[task_key]
            
            # Check for errors
            if result.get('error'):
                print(f"✗ Error for {task_key}: {result['error']}")
                task.failed = True
                task.error_message = result['error']
                continue
            
            # Extract response text
            model_output = result.get('modelOutput', {})
            content = model_output.get('content', [])
            
            if not content:
                print(f"✗ Empty response for {task_key}")
                continue
            
            response_text = content[0].get('text', '')
            
            # Track API call
            task.total_api_calls += 1
            
            # Process based on stage
            if stage == "agent1":
                task.agent1_scenario = response_text
                task.agent1_iteration += 1
                task.current_stage = "agent2"
                task.set_next_stage("agent2")
                print(f"✓ {task_key}: Agent 1 complete (iteration {task.agent1_iteration})")
            
            elif stage == "agent2":
                # Extract JSON from response
                try:
                    json_text = self._extract_json_from_response(response_text)
                    task.agent2_structured_json = json.loads(json_text)
                    task.agent2_iteration += 1
                    task.current_stage = "judge1"
                    task.set_next_stage("judge1")
                    print(f"✓ {task_key}: Agent 2 complete (iteration {task.agent2_iteration})")
                except Exception as e:
                    print(f"✗ {task_key}: Failed to parse Agent 2 output: {e}")
                    task.mark_failed(str(e))
            
            elif stage == "judge1":
                # Parse judge decision
                try:
                    json_text = self._extract_json_from_response(response_text)
                    judgment = json.loads(json_text)
                    
                    approved = judgment.get("approved", False)
                    task.judge1_approved = approved
                    
                    if approved:
                        task.current_stage = "agent3"
                        task.set_next_stage("agent3")
                        print(f"✓ {task_key}: Judge 1 approved")
                    else:
                        # Determine if Agent 1 or Agent 2 needs to retry
                        error_source = judgment.get('error_source', 'agent2')
                        
                        # Check iteration limits (matching demo_sts_sde.py logic)
                        if error_source == 'agent1':
                            # Check outer iteration limit (Agent 1 regeneration)
                            if task.agent1_iteration >= self.judge1_max_outer_iterations:
                                print(f"⚠ {task_key}: Agent 1 reached max iterations ({self.judge1_max_outer_iterations}), using last result")
                                task.judge1_approved = True
                                task.current_stage = "agent3"
                                task.set_next_stage("agent3")
                                continue
                        elif error_source == 'agent2':
                            # Check inner iteration limit (Agent 2 parsing)
                            if task.agent2_iteration >= self.judge1_max_inner_iterations:
                                print(f"⚠ {task_key}: Agent 2 reached max iterations ({self.judge1_max_inner_iterations}), escalating to Agent 1")
                                error_source = 'agent1'  # Escalate to Agent 1 if Agent 2 repeatedly fails
                        
                        feedback = self._format_feedback_for_agent1(judgment) if error_source == 'agent1' else self._format_feedback_for_agent2(judgment)
                        
                        task.current_stage = error_source
                        task.set_next_stage(error_source, feedback)
                        print(f"⚠ {task_key}: Judge 1 rejected, retry {error_source} (iter: Agent1={task.agent1_iteration}/{self.judge1_max_outer_iterations}, Agent2={task.agent2_iteration}/{self.judge1_max_inner_iterations})")
                except Exception as e:
                    print(f"✗ {task_key}: Failed to parse Judge 1 output: {e}")
                    # Default to approval on error
                    task.judge1_approved = True
                    task.current_stage = "agent3"
                    task.set_next_stage("agent3")
            
            elif stage == "agent3":
                try:
                    json_text = self._extract_json_from_response(response_text)
                    task.agent3_sde_params = json.loads(json_text)
                    task.current_stage = "agent4"
                    task.set_next_stage("agent4")
                    print(f"✓ {task_key}: Agent 3 complete")
                except Exception as e:
                    print(f"✗ {task_key}: Failed to parse Agent 3 output: {e}")
                    task.mark_failed(str(e))
            
            elif stage == "agent4":
                try:
                    json_text = self._extract_json_from_response(response_text)
                    task.agent4_adjacency = json.loads(json_text)
                    task.agent34_iteration += 1
                    task.current_stage = "judge2"
                    task.set_next_stage("judge2")
                    print(f"✓ {task_key}: Agent 4 complete, moving to Judge 2")
                except Exception as e:
                    print(f"✗ {task_key}: Failed to parse Agent 4 output: {e}")
                    task.mark_failed(str(e))
            
            elif stage == "judge2":
                # Parse judge 2 decision
                try:
                    json_text = self._extract_json_from_response(response_text)
                    judgment = json.loads(json_text)
                    
                    approved = judgment.get("approved", False)
                    task.judge2_approved = approved
                    
                    if approved:
                        task.mark_complete()
                        print(f"✓ {task_key}: Judge 2 approved - TASK FINISHED")
                    else:
                        # Check iteration limit for Agent 3&4 (Judge 2)
                        if task.agent34_iteration >= self.judge2_max_iterations:
                            print(f"⚠ {task_key}: Agent 3&4 reached max iterations ({self.judge2_max_iterations}), using last result")
                            task.judge2_approved = True
                            task.mark_complete()
                        else:
                            # Need to regenerate Agent 3 and 4
                            # Format feedback
                            feedback_text = f"OVERALL: {judgment.get('overall_comment', '')}\n"
                            feedback_text += f"\nVISUAL ASSESSMENT: {judgment.get('visual_assessment', '')}\n"
                            
                            param_issues = judgment.get('parameter_issues', [])
                            if param_issues:
                                feedback_text += f"\nPARAMETER ISSUES:\n"
                                for issue in param_issues:
                                    feedback_text += f"- Node {issue.get('node_id')}: {issue.get('problem')}\n"
                                    feedback_text += f"  Suggested: {issue.get('suggested_value')}\n"
                            
                            adj_issues = judgment.get('adjacency_issues', [])
                            if adj_issues:
                                feedback_text += f"\nADJACENCY ISSUES:\n"
                                for issue in adj_issues:
                                    feedback_text += f"- Edge {issue.get('edge')}: {issue.get('problem')}\n"
                                    feedback_text += f"  Suggestion: {issue.get('suggestion')}\n"
                            
                            task.current_stage = "agent3"
                            task.set_next_stage("agent3", feedback_text)
                            print(f"⚠ {task_key}: Judge 2 rejected, regenerating parameters (iter: {task.agent34_iteration}/{self.judge2_max_iterations})")
                except Exception as e:
                    print(f"✗ {task_key}: Failed to parse Judge 2 output: {e}")
                    # Default to approval on error
                    task.judge2_approved = True
                    task.mark_complete()
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON content from response text."""
        import re
        
        text = response_text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
        
        # Find JSON object
        brace_count = 0
        start_idx = -1
        end_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    end_idx = i
                    break
        
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx+1].strip()
        
        return text.strip()
    
    def _format_feedback_for_agent1(self, judgment: Dict) -> str:
        """Format feedback for Agent 1."""
        feedback = f"SCENARIO LOGIC ISSUES:\n\n"
        feedback += f"Overall: {judgment.get('feedback', '')}\n\n"
        
        for idx, issue in enumerate(judgment.get('issues', []), 1):
            if issue.get('type') == 'Scenario Logic':
                feedback += f"\n{idx}. {issue.get('field', 'N/A')}\n"
                feedback += f"   Problem: {issue.get('problem', 'N/A')}\n"
                feedback += f"   Suggestion: {issue.get('suggestion', 'N/A')}\n"
        
        return feedback
    
    def _format_feedback_for_agent2(self, judgment: Dict) -> str:
        """Format feedback for Agent 2."""
        feedback = f"PARSING FIDELITY ISSUES:\n\n"
        feedback += f"Overall: {judgment.get('feedback', '')}\n\n"
        
        for idx, issue in enumerate(judgment.get('issues', []), 1):
            if issue.get('type') == 'Parsing Fidelity':
                feedback += f"\n{idx}. {issue.get('field', 'N/A')}\n"
                feedback += f"   Problem: {issue.get('problem', 'N/A')}\n"
                feedback += f"   Suggestion: {issue.get('suggestion', 'N/A')}\n"
        
        return feedback
    
    def _wait_for_rate_limit(self):
        """Wait to respect rate limiting for real-time API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_realtime_request_time
        
        if time_since_last_request < self.realtime_request_interval:
            wait_time = self.realtime_request_interval - time_since_last_request
            print(f"    ⏱️  Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        self.last_realtime_request_time = time.time()
    
    def _execute_realtime_inference(self, task: TaskState, retry_count: int = 0) -> Tuple[bool, str]:
        """
        Execute a single real-time API call for a task.
        
        Returns:
            Tuple[bool, str]: (success, response_text or error_message)
        """
        stage = task.next_stage
        
        try:
            # Wait for rate limiting
            self._wait_for_rate_limit()
            
            # Build request based on stage
            print(f"      🔄 Calling LLM API ({stage})...")
            if stage == "judge2":
                print(f"      📊 Generating visualization for Judge 2...")
                image_base64 = self._generate_preview_visualization(task)
                prompt = self._create_judge2_prompt(task)
                response_text = self.llm.complete_with_image(
                    prompt=prompt,
                    image_base64=image_base64,
                )
            else:
                prompt = getattr(self, f"_create_{stage}_prompt")(task)
                response_text = self.llm.complete(prompt=prompt)

            return True, response_text

        except Exception as e:
            print(f"      ❌ Error: {e}")
            return False, str(e)
    
    def _process_realtime_result(self, task: TaskState, task_key: str, stage: str, response_text: str):
        """Process a single real-time result and update task state."""
        try:
            task.total_api_calls += 1
            
            # Process based on stage (reuse batch processing logic)
            if stage == "agent1":
                task.agent1_scenario = response_text
                task.agent1_iteration += 1
                task.current_stage = "agent2"
                task.set_next_stage("agent2")
                print(f"      ✓ Agent 1 complete (iteration {task.agent1_iteration})")
            
            elif stage == "agent2":
                # Extract JSON from response
                json_text = self._extract_json_from_response(response_text)
                task.agent2_structured_json = json.loads(json_text)
                task.agent2_iteration += 1
                task.current_stage = "judge1"
                task.set_next_stage("judge1")
                print(f"      ✓ Agent 2 complete (iteration {task.agent2_iteration})")
            
            elif stage == "judge1":
                # Parse judge decision
                json_text = self._extract_json_from_response(response_text)
                judgment = json.loads(json_text)
                
                approved = judgment.get("approved", False)
                task.judge1_approved = approved
                
                if approved:
                    task.current_stage = "agent3"
                    task.set_next_stage("agent3")
                    print(f"      ✓ Judge 1 approved")
                else:
                    # Determine if Agent 1 or Agent 2 needs to retry
                    error_source = judgment.get('error_source', 'agent2')
                    
                    # Check iteration limits
                    if error_source == 'agent1':
                        if task.agent1_iteration >= self.judge1_max_outer_iterations:
                            print(f"      ⚠️  Agent 1 max iterations reached, forcing approval")
                            task.judge1_approved = True
                            task.current_stage = "agent3"
                            task.set_next_stage("agent3")
                            return
                    elif error_source == 'agent2':
                        if task.agent2_iteration >= self.judge1_max_inner_iterations:
                            print(f"      ⚠️  Agent 2 max iterations reached, escalating to Agent 1")
                            error_source = 'agent1'
                    
                    feedback = self._format_feedback_for_agent1(judgment) if error_source == 'agent1' else self._format_feedback_for_agent2(judgment)
                    next_stage = 'agent1' if error_source == 'agent1' else 'agent2'
                    task.current_stage = next_stage
                    task.set_next_stage(next_stage, feedback)
                    print(f"      ⚠️  Judge 1 rejected, retry {next_stage}")
            
            elif stage == "agent3":
                json_text = self._extract_json_from_response(response_text)
                task.agent3_sde_params = json.loads(json_text)
                task.current_stage = "agent4"
                task.set_next_stage("agent4")
                print(f"      ✓ Agent 3 complete")
            
            elif stage == "agent4":
                json_text = self._extract_json_from_response(response_text)
                task.agent4_adjacency = json.loads(json_text)
                task.agent34_iteration += 1
                task.current_stage = "judge2"
                task.set_next_stage("judge2")
                print(f"      ✓ Agent 4 complete (iteration {task.agent34_iteration})")
            
            elif stage == "judge2":
                json_text = self._extract_json_from_response(response_text)
                judgment = json.loads(json_text)
                
                approved = judgment.get("approved", False)
                task.judge2_approved = approved
                
                if approved:
                    task.mark_complete()
                    print(f"      ✓ Judge 2 approved - TASK COMPLETE!")
                else:
                    # Check iteration limit
                    if task.agent34_iteration >= self.judge2_max_iterations:
                        print(f"      ⚠️  Judge 2 max iterations reached, forcing completion")
                        task.mark_complete()
                        return
                    
                    # Format feedback and retry Agent 3&4
                    feedback_text = f"OVERALL: {judgment.get('overall_comment', '')}\n"
                    feedback_text += f"VISUAL: {judgment.get('visual_assessment', '')}\n"
                    
                    for issue in judgment.get('parameter_issues', []):
                        feedback_text += f"- Node {issue.get('node_id')}: {issue.get('problem')}\n"
                    
                    task.judge2_feedback = feedback_text
                    task.current_stage = "agent3"
                    task.set_next_stage("agent3", feedback_text)
                    print(f"      ⚠️  Judge 2 rejected, retry Agent 3&4")
        
        except Exception as e:
            print(f"      ❌ Error processing result: {e}")
            task.mark_failed(str(e))
    
    def _process_remaining_tasks_realtime(self, ready_tasks: List[Tuple[str, TaskState]]):
        """
        Process remaining tasks using real-time API with rate limiting and retry logic.
        """
        print(f"\n{'='*80}")
        print(f"REAL-TIME PROCESSING MODE ({len(ready_tasks)} tasks)")
        print(f"{'='*80}\n")
        
        for idx, (task_key, task) in enumerate(ready_tasks, 1):
            stage = task.next_stage
            print(f"  [{idx}/{len(ready_tasks)}] Processing {task_key} → {stage}")
            
            task.mark_processing()
            
            # Execute real-time inference with retry logic
            success, result = self._execute_realtime_inference(task)
            
            if success:
                # Process the result
                self._process_realtime_result(task, task_key, stage, result)
            else:
                # Failed after retries - mark for requeue
                print(f"      ⚠️  Failed, will retry in next iteration: {result}")
                task.is_ready = True
                task.is_processing = False
        
        print(f"\n{'='*80}")
        print(f"REAL-TIME BATCH COMPLETE")
        print(f"{'='*80}\n")
    
    def run_queue_driven_pipeline(self, batch_size: int = 100, max_total_iterations: int = 50):
        """
        Run the queue-driven pipeline: submit mixed batches of up to batch_size tasks at once.
        This is more efficient than stage-by-stage processing.
        """
        print("\n" + "="*80)
        print("QUEUE-DRIVEN BATCH INFERENCE PIPELINE")
        print("="*80)
        print(f"Batch size: {batch_size}")
        print(f"Max iterations: {max_total_iterations}")
        print("="*80 + "\n")
        
        iteration = 0
        
        while iteration < max_total_iterations:
            iteration += 1
            
            # Check completion status
            ready_count = sum(1 for t in self.tasks.values() if t.is_ready and not t.completed and not t.failed)
            completed = sum(1 for t in self.tasks.values() if t.completed)
            failed = sum(1 for t in self.tasks.values() if t.failed)
            processing = sum(1 for t in self.tasks.values() if t.is_processing)
            
            # Print progress
            self.progress_monitor.print_progress(self.tasks, force=(iteration % 5 == 0))
            
            # Check if all tasks are done
            if ready_count == 0 and processing == 0:
                print(f"\n{'='*80}")
                print("✓ ALL TASKS COMPLETED!")
                print(f"{'='*80}")
                print(f"Final Status:")
                print(f"  ✅ Completed: {completed}/{len(self.tasks)}")
                print(f"  ❌ Failed: {failed}/{len(self.tasks)}")
                break
            
            if ready_count == 0:
                print(f"\nIteration {iteration}: No ready tasks, all {processing} tasks are still processing. Waiting...")
                time.sleep(30)
                continue
            
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration}/{max_total_iterations}")
            print(f"{'='*80}")
            print(f"Ready tasks: {ready_count} | Processing: {processing} | Completed: {completed} | Failed: {failed}")
            
            # HYBRID MODEL: Switch between batch and real-time processing
            if 0 < ready_count < self.switch_to_realtime_threshold:
                # REAL-TIME MODE for small number of remaining tasks
                print(f"\n🔄 Switching to REAL-TIME mode ({ready_count} < {self.switch_to_realtime_threshold} tasks)")
                ready_tasks = self.get_ready_tasks(limit=ready_count)
                self._process_remaining_tasks_realtime(ready_tasks)
                time.sleep(2)  # Short pause before next iteration
                continue
            
            # BATCH MODE for large number of tasks
            print(f"\n📦 Using BATCH mode ({ready_count} tasks)")
            
            # Get up to batch_size ready tasks
            ready_tasks = self.get_ready_tasks(limit=batch_size)
            
            if not ready_tasks:
                print("No ready tasks, waiting...")
                time.sleep(30)
                continue
            
            print(f"\nSubmitting batch of {len(ready_tasks)} tasks...")
            
            # Show stage distribution
            stage_dist = {}
            for _, task in ready_tasks:
                stage = task.next_stage
                stage_dist[stage] = stage_dist.get(stage, 0) + 1
            
            print(f"Stage distribution:")
            for stage, count in sorted(stage_dist.items()):
                print(f"  {stage}: {count} tasks")
            
            # Create mixed batch requests
            requests = self.create_mixed_batch_requests(ready_tasks)
            
            if not requests:
                print("Failed to create requests, skipping...")
                continue
            
            # Pad if necessary
            requests = self._pad_batch_to_minimum(requests, min_size=100)
            
            # Submit batch job
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_name_stage = f"mixed-batch-iter{iteration}"  # Use hyphens instead of underscores for AWS compatibility
            job_arn, output_uri = self.submit_batch_job(requests, job_name_stage)
            
            if not job_arn:
                # Mark all processing tasks as ready again on failure
                for _, task in ready_tasks:
                    task.is_ready = True
                    task.is_processing = False
                continue
            
            # Wait for completion
            self.wait_for_job(job_arn)
            
            # Download and process results
            results = self.download_results(output_uri)
            
            # Process results for mixed stages (need to determine stage from recordId)
            self._process_mixed_results(results)
            
            # Save checkpoint
            self.save_checkpoint()
        
        print("\n" + "="*80)
        print("QUEUE-DRIVEN PIPELINE COMPLETE")
        print("="*80)
        
        # Final progress report
        self.progress_monitor.print_progress(self.tasks, force=True)
        
        self.save_final_results()
    
    def _process_mixed_results(self, results: List[Dict]):
        """Process results from a mixed batch (tasks at different stages)."""
        print(f"\nProcessing {len(results)} mixed batch results...")
        
        # Filter out padding
        real_results = []
        for r in results:
            record_id = r.get('recordId') or r.get('record_id', '')
            if '_pad' not in record_id:
                real_results.append(r)
        
        print(f"Real results (non-padded): {len(real_results)}")
        
        # Group by stage for processing
        results_by_stage = {}
        for result in real_results:
            record_id = result.get('recordId') or result.get('record_id', '')
            
            # Extract stage from recordId (e.g., "task_0001_agent1_iter0" -> "agent1")
            if '_agent1_' in record_id:
                stage = "agent1"
            elif '_agent2_' in record_id:
                stage = "agent2"
            elif '_agent3_' in record_id:
                stage = "agent3"
            elif '_agent4_' in record_id:
                stage = "agent4"
            elif '_judge1_' in record_id:
                stage = "judge1"
            elif '_judge2_' in record_id:
                stage = "judge2"
            else:
                print(f"Warning: Cannot determine stage from recordId: {record_id}")
                continue
            
            if stage not in results_by_stage:
                results_by_stage[stage] = []
            results_by_stage[stage].append(result)
        
        # Process each stage's results
        for stage, stage_results in results_by_stage.items():
            print(f"\n  Processing {len(stage_results)} results for stage: {stage}")
            self.process_results(stage_results, stage)
    
    def run_pipeline(self, max_iterations: int = 3):
        """Run the complete multi-stage pipeline (old stage-by-stage method, kept for compatibility)."""
        print("\n" + "="*80)
        print("BATCH INFERENCE PIPELINE START (Stage-by-Stage Mode)")
        print("="*80)
        print("Note: Consider using run_queue_driven_pipeline() for better efficiency")
        print("="*80 + "\n")
        
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'='*80}")
            
            # Check completion status
            completed = sum(1 for t in self.tasks.values() if t.completed)
            failed = sum(1 for t in self.tasks.values() if t.failed)
            in_progress = len(self.tasks) - completed - failed
            
            print(f"\nTask Status:")
            print(f"  ✓ Completed: {completed}/{len(self.tasks)}")
            print(f"  ✗ Failed: {failed}/{len(self.tasks)}")
            print(f"  ⟳ In Progress: {in_progress}/{len(self.tasks)}")
            
            if in_progress == 0:
                print("\n✓ All tasks completed!")
                break
            
            # Process each stage in sequence
            for stage in ["agent1", "agent2", "judge1", "agent3", "agent4", "judge2"]:
                requests = self.create_batch_requests(stage)
                
                if not requests:
                    print(f"  No tasks for {stage}")
                    continue
                
                print(f"\n  Processing {stage}: {len(requests)} tasks")
                
                job_arn, output_uri = self.submit_batch_job(requests, stage)
                if not job_arn:
                    continue
                
                # Wait for completion
                self.wait_for_job(job_arn)
                
                # Download and process results
                results = self.download_results(output_uri)
                self.process_results(results, stage)
                
                # Save checkpoint
                self.save_checkpoint()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        
        self.save_final_results()
    
    def save_checkpoint(self):
        """Save current state to file."""
        checkpoint_file = os.path.join(OUTPUT_DIR, "checkpoint.pkl")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self.tasks, f)
        
        # Also save human-readable summary
        summary_file = os.path.join(OUTPUT_DIR, "task_summary.json")
        summary = {k: v.to_dict() for k, v in self.tasks.items()}
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def save_final_results(self):
        """Save final results for all completed tasks in multiple formats (JSON, PKL, TXT, HTML)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        completed_tasks = [t for t in self.tasks.values() if t.completed]
        total_completed = len(completed_tasks)
        
        print(f"\n{'='*80}")
        print(f"SAVING FINAL RESULTS FOR {total_completed} COMPLETED TASKS")
        print(f"{'='*80}")
        
        saved_files = []
        
        for idx, (task_key, task) in enumerate(self.tasks.items(), 1):
            if not task.completed:
                continue
            
            print(f"\n[{idx}/{total_completed}] {task_key} ({task.domain}, {task.num_nodes} nodes)")
            
            try:
                # Generate complete results (JSON, PKL, TXT, HTML)
                files = self._generate_complete_results(task, task_key, timestamp)
                saved_files.append({
                    "task_key": task_key,
                    "domain": task.domain,
                    "num_nodes": task.num_nodes,
                    "files": files
                })
            except Exception as e:
                print(f"    ✗ Error generating results for {task_key}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Total completed tasks: {total_completed}")
        print(f"Successfully saved: {len(saved_files)}")
        print(f"Failed: {total_completed - len(saved_files)}")
        
        # Create summary file
        summary_file = os.path.join(OUTPUT_DIR, f"batch_summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BATCH INFERENCE RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total tasks: {len(self.tasks)}\n")
            f.write(f"Completed: {total_completed}\n")
            f.write(f"Failed: {sum(1 for t in self.tasks.values() if t.failed)}\n")
            f.write(f"In progress: {len(self.tasks) - total_completed - sum(1 for t in self.tasks.values() if t.failed)}\n\n")
            
            f.write("Completed Tasks:\n")
            f.write("-"*80 + "\n")
            for item in saved_files:
                f.write(f"\n{item['task_key']} ({item['domain']}, {item['num_nodes']} nodes)\n")
                for file_type, file_path in item['files'].items():
                    f.write(f"  {file_type.upper()}: {os.path.basename(file_path)}\n")
        
        print(f"\nBatch summary saved: {summary_file}")
        print(f"Output directory: {OUTPUT_DIR}/")
        print(f"{'='*80}\n")
    
    def _generate_complete_results(self, task: TaskState, task_key: str, timestamp: str):
        """Generate complete results including JSON, HTML, TXT, and PKL files."""
        import numpy as np
        import pickle
        from demo_sts_sde import NetworkSDEGenerator
        
        print(f"\n  Generating complete results for {task_key}...")
        
        # Extract sequence length from structured scenario
        seq_len = self._calculate_seq_len_from_scenario(task.agent2_structured_json)
        
        # Create temporary generator to simulate data
        temp_gen = NetworkSDEGenerator(num_nodes=task.num_nodes)
        temp_gen.seq_len = seq_len
        temp_gen.sde_params = task.agent3_sde_params
        
        # Extract edge lags from structured scenario
        edge_lags = {}
        for edge in task.agent2_structured_json.get('edges', []):
            if 'time_lag' in edge:
                edge_key = f"{edge['source']}->{edge['target']}"
                edge_lags[edge_key] = edge['time_lag']
        
        # Build network_sde structure
        network_sde = {
            "structured_scenario": task.agent2_structured_json,
            "sequence_length": seq_len,
            "sde_parameters": task.agent3_sde_params,
            "time_varying_adjacency": task.agent4_adjacency,
            "dt": 1.0,
            "noise_correlation": np.eye(task.num_nodes),
            "edge_lags": edge_lags
        }
        
        # Generate time series data
        ts_data, generation_info = temp_gen.generate_spatiotemporal_data(network_sde)
        
        # Add domain and task_id to structured scenario for display
        network_sde["structured_scenario"]["domain"] = task.domain
        network_sde["structured_scenario"]["task_id"] = task_key
        
        # 1. Save JSON
        json_file = os.path.join(OUTPUT_DIR, f"{task_key}_{task.domain}_{timestamp}.json")
        json_data = {
            "task_id": task.task_id,
            "num_nodes": task.num_nodes,
            "domain": task.domain,
            "agent1_scenario": task.agent1_scenario,
            "agent2_structured_scenario": task.agent2_structured_json,
            "agent3_sde_parameters": task.agent3_sde_params,
            "agent4_time_varying_adjacency": task.agent4_adjacency,
            "agent5_simulation_data": ts_data.tolist(),
            "iterations": {
                "agent1": task.agent1_iteration,
                "agent2": task.agent2_iteration,
                "agent34": task.agent34_iteration
            },
            "seq_len": seq_len,
            "timestamp": timestamp,
            "config": {
                "num_nodes": task.num_nodes,
                "domain": task.domain,
                "sequence_length": seq_len,
                "time_span": task.agent2_structured_json.get("time_span", "unknown"),
                "sampling_frequency": task.agent2_structured_json.get("sampling_frequency", "unknown"),
                "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                "method": "openai-compatible-llm-api"
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"    ✓ JSON: {json_file}")
        
        # 2. Save PKL (complete Python objects)
        pkl_file = os.path.join(OUTPUT_DIR, f"{task_key}_{task.domain}_{timestamp}.pkl")
        pkl_data = {
            "timestamp": timestamp,
            "task_id": task.task_id,
            "domain": task.domain,
            "agent1_scenario": task.agent1_scenario,
            "agent2_structured_scenario": task.agent2_structured_json,
            "agent3_sde_parameters": task.agent3_sde_params,
            "agent4_time_varying_adjacency": task.agent4_adjacency,
            "agent5_simulation_data": ts_data,
            "generation_info": generation_info,
            "seq_len": seq_len
        }
        
        with open(pkl_file, 'wb') as f:
            pickle.dump(pkl_data, f)
        print(f"    ✓ PKL: {pkl_file}")
        
        # 3. Save TXT (human-readable description)
        txt_file = os.path.join(OUTPUT_DIR, f"{task_key}_{task.domain}_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"Spatial-Temporal Data Generation Results (Batch API)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Task ID: {task_key}\n")
            f.write(f"Domain: {task.domain}\n")
            f.write(f"Generation time: {timestamp}\n")
            f.write(f"Iterations: Agent1={task.agent1_iteration}, Agent2={task.agent2_iteration}, Agent3/4={task.agent34_iteration}\n\n")
            
            f.write("Agent 1: Scenario Description\n")
            f.write("-" * 80 + "\n")
            f.write(task.agent1_scenario + "\n\n")
            
            f.write("Agent 2: Structured Scenario\n")
            f.write("-" * 80 + "\n")
            f.write(f"Variable: {task.agent2_structured_json.get('variable', 'N/A')}\n")
            f.write(f"Time span: {task.agent2_structured_json.get('time_span', 'N/A')}\n")
            f.write(f"Sampling frequency: {task.agent2_structured_json.get('sampling_frequency', 'N/A')}\n")
            f.write(f"Sequence length: {seq_len}\n")
            f.write(f"Number of nodes: {task.num_nodes}\n\n")
            
            f.write("Nodes:\n")
            for node in task.agent2_structured_json.get('nodes', []):
                f.write(f"  Node {node['id']}: [{node['type']}] {node.get('name', 'N/A')}\n")
                f.write(f"    {node.get('description', '')}\n")
            
            f.write("\nEdges:\n")
            for edge in task.agent2_structured_json.get('edges', []):
                f.write(f"  {edge['source']} -> {edge['target']}: {edge.get('relationship', 'N/A')}\n")
                if 'time_lag' in edge:
                    f.write(f"    Time lag: {edge['time_lag']} steps\n")
            
            f.write("\n")
            f.write("Agent 5: Simulation Results\n")
            f.write("-" * 80 + "\n")
            f.write(f"Integration method: {generation_info.get('integration_method', 'N/A')}\n")
            f.write(f"Time step (dt): {generation_info.get('dt', 1.0):.6f}\n")
            f.write(f"Generated data shape: {ts_data.shape}\n")
            f.write(f"Data statistics:\n")
            for i in range(task.num_nodes):
                node_name = task.agent2_structured_json['nodes'][i].get('name', f'Node {i}')
                f.write(f"  {node_name}: min={ts_data[i].min():.2f}, max={ts_data[i].max():.2f}, mean={ts_data[i].mean():.2f}\n")
        
        print(f"    ✓ TXT: {txt_file}")
        
        # 4. Save HTML (interactive visualization)
        html_file = os.path.join(OUTPUT_DIR, f"{task_key}_{task.domain}_interactive_{timestamp}.html")
        temp_gen.visualize_network_sde_html(ts_data, network_sde, generation_info, save_path=html_file)
        print(f"    ✓ HTML: {html_file}")
        
        return {
            "json": json_file,
            "pkl": pkl_file,
            "txt": txt_file,
            "html": html_file
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch inference runner for STS generation')
    parser.add_argument('--min_tasks', type=int, default=100,
                       help='Minimum number of tasks to generate (default: 100)')
    parser.add_argument('--max_iterations', type=int, default=30,
                       help='Maximum total iterations for the queue-driven pipeline (default: 30)')
    parser.add_argument('--node_counts', type=str, default='3,5,10',
                       help='Comma-separated list of node counts (default: 3,5,10)')
    
    # Judge Agent iteration limits (matching demo_sts_sde.py)
    parser.add_argument('--judge1_max_outer', type=int, default=3,
                       help='Max outer iterations for Judge 1 (Agent 1 regeneration, default: 3)')
    parser.add_argument('--judge1_max_inner', type=int, default=2,
                       help='Max inner iterations for Judge 1 (Agent 2 parsing, default: 2)')
    parser.add_argument('--judge2_max_iter', type=int, default=5,
                       help='Max iterations for Judge 2 (Agent 3&4 regeneration, default: 5)')
    
    # Hybrid execution model
    parser.add_argument('--realtime_threshold', type=int, default=10,
                       help='Switch to real-time API when tasks < threshold (default: 10)')
    
    args = parser.parse_args()
    
    # Configuration from run_experiments.sh
    domains = [
        "Transportation",
        "Energy",
        "Environment&Pollution",
        "Ecology",
        "Public Health",
        "Hydrology",
        "Oceanography",
        "Agriculture",
        "Mobility",
        "Climate"
    ]
    
    # Parse node counts
    node_counts = [int(n.strip()) for n in args.node_counts.split(',')]
    
    print("\n" + "="*80)
    print("BATCH INFERENCE CONFIGURATION")
    print("="*80)
    print(f"Independent samples to generate: {args.min_tasks}")
    print(f"Node counts: {node_counts}")
    print(f"Domains: {len(domains)} domains")
    print(f"Max iterations per task: {args.max_iterations}")
    print(f"Unique domain×node combinations: {len(node_counts) * len(domains)}")
    
    samples_per_combo = args.min_tasks // (len(node_counts) * len(domains))
    if samples_per_combo > 1:
        print(f"Samples per combination: ~{samples_per_combo} (each is an independent data generation)")
    
    print("\nNote: All tasks are real independent samples requiring unique data.")
    print("Padding is only used when a stage has < 100 tasks (AWS Batch API requirement).")
    print("="*80 + "\n")
    
    # Create runner with Judge Agent iteration limits and hybrid execution model
    runner = BatchInferenceRunner(
        domains, 
        node_counts, 
        min_tasks=args.min_tasks,
        judge1_max_outer_iterations=args.judge1_max_outer,
        judge1_max_inner_iterations=args.judge1_max_inner,
        judge2_max_iterations=args.judge2_max_iter,
        switch_to_realtime_threshold=args.realtime_threshold
    )
    
    # Run queue-driven pipeline (new efficient method)
    runner.run_queue_driven_pipeline(batch_size=100, max_total_iterations=args.max_iterations)


if __name__ == "__main__":
    main()

