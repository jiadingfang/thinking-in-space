#!/bin/bash

set -e

# Configuration
# Get API key from environment variable - set this before running the script
# export GOOGLE_API_KEY="your_api_key_here"
export LMMS_EVAL_LAUNCHER="python"

# Default settings
BENCHMARK="vsibench"
OUTPUT_PATH="logs/$(TZ="America/New_York" date "+%Y%m%d")"
RATIO=0.1  # Default 10% from each task
NUM_PROCESSES=1

# Available Gemini models
AVAILABLE_MODELS=("gemini_1p5_flash" "gemini_1p5_pro_002" "gemini_2p0_flash_exp" "gemini_2p5_pro" "gemini_2p5_flash" "gemini_2p5_flash_lite_preview")

# VSI-Bench task types
MCA_TASKS=("object_rel_direction_easy" "object_rel_direction_medium" "object_rel_direction_hard" "object_rel_distance" "route_planning" "obj_appearance_order")
NA_TASKS=("object_abs_distance" "object_counting" "object_size_estimation" "room_size_estimation")
ALL_TASKS=("${MCA_TASKS[@]}" "${NA_TASKS[@]}")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        --num_processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --list-models)
            echo "Available Gemini models:"
            for model in "${AVAILABLE_MODELS[@]}"; do
                echo "  - $model"
            done
            exit 0
            ;;
        --list-tasks)
            echo "Available VSI-Bench task types:"
            echo "Multiple Choice Answer (MCA) tasks:"
            for task in "${MCA_TASKS[@]}"; do
                echo "  - $task"
            done
            echo "Numerical Answer (NA) tasks:"
            for task in "${NA_TASKS[@]}"; do
                echo "  - $task"
            done
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Gemini model to evaluate (default: gemini_1p5_flash)"
            echo "  --ratio RATIO          Ratio of samples per task (0.0-1.0, default: 0.1)"
            echo "  --num_processes N      Number of processes (default: 1)"
            echo "  --benchmark BENCHMARK  Benchmark to run (default: vsibench)"
            echo "  --output_path PATH     Output path (default: logs/YYYYMMDD)"
            echo "  --list-models          List available models"
            echo "  --list-tasks           List available task types"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model gemini_1p5_flash --ratio 0.1"
            echo "  $0 --model gemini_2p5_flash --ratio 0.2"
            echo "  $0 --list-tasks"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default model if not specified
if [ -z "$MODEL" ]; then
    MODEL="gemini_1p5_flash"
fi

# Validate model
MODEL_VALID=false
for valid_model in "${AVAILABLE_MODELS[@]}"; do
    if [ "$MODEL" = "$valid_model" ]; then
        MODEL_VALID=true
        break
    fi
done

if [ "$MODEL_VALID" = false ]; then
    echo "Error: Invalid model '$MODEL'"
    echo "Available models: ${AVAILABLE_MODELS[*]}"
    exit 1
fi

# Validate ratio
if ! [[ "$RATIO" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "$RATIO <= 0" | bc -l) )) || (( $(echo "$RATIO > 1" | bc -l) )); then
    echo "Error: Ratio must be between 0.0 and 1.0"
    exit 1
fi

# Check if Google API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: GOOGLE_API_KEY environment variable is not set"
    echo ""
    echo "Please set your Google API key before running this script:"
    echo "  export GOOGLE_API_KEY=\"your_api_key_here\""
    echo ""
    echo "Or add it to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    echo "  echo 'export GOOGLE_API_KEY=\"your_api_key_here\"' >> ~/.zshrc"
    echo "  source ~/.zshrc"
    echo ""
    echo "You can get your API key from: https://makersuite.google.com/app/apikey"
    exit 1
fi

# Check if vsibench environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "vsibench" ]; then
    echo "Warning: vsibench conda environment is not activated"
    echo "Please run: conda activate vsibench"
    echo "Continuing anyway..."
fi

echo "=== Balanced Gemini Model Evaluation ==="
echo "Model: $MODEL"
echo "Benchmark: $BENCHMARK"
echo "Ratio per task: $RATIO (${RATIO}%)"
echo "Total tasks: ${#ALL_TASKS[@]}"
echo "Output: $OUTPUT_PATH/$BENCHMARK"
echo "========================================"

# Set model configuration based on model name
case "$MODEL" in
    "gemini_1p5_flash")
        MODEL_FAMILY="gemini_api"
        MODEL_ARGS="model_version=gemini-1.5-flash,modality=video"
        ;;
    "gemini_1p5_pro_002")
        MODEL_FAMILY="gemini_api"
        MODEL_ARGS="model_version=gemini-1.5-pro,modality=video"
        ;;
    "gemini_2p0_flash_exp")
        MODEL_FAMILY="gemini_api"
        MODEL_ARGS="model_version=gemini-2.0-flash-exp,modality=video"
        ;;
    "gemini_2p5_pro")
        MODEL_FAMILY="gemini_api"
        MODEL_ARGS="model_version=gemini-2.5-pro,modality=video"
        ;;
    "gemini_2p5_flash")
        MODEL_FAMILY="gemini_api"
        MODEL_ARGS="model_version=gemini-2.5-flash,modality=video"
        ;;
    "gemini_2p5_flash_lite_preview")
        MODEL_FAMILY="gemini_api"
        MODEL_ARGS="model_version=gemini-2.5-flash-lite-preview-06-17,modality=video"
        ;;
    *)
        echo "Error: Unknown model configuration for $MODEL"
        exit 1
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_PATH/$BENCHMARK"

# Create balanced sampling script
BALANCED_SCRIPT=$(cat << 'EOF'
import os
import json
import random
from datasets import load_dataset
import pandas as pd

def create_balanced_sample(ratio=0.1, seed=42):
    """Create a balanced sample from VSI-Bench with equal representation from each task type."""
    
    # Load the full dataset
    print("Loading VSI-Bench dataset...")
    dataset = load_dataset("nyu-visionx/VSI-Bench")
    test_data = dataset["test"]
    
    # Define task types
    mca_tasks = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard", 
                 "object_rel_distance", "route_planning", "obj_appearance_order"]
    na_tasks = ["object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"]
    all_tasks = mca_tasks + na_tasks
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    balanced_indices = []
    task_counts = {}
    
    print(f"Creating balanced sample with {ratio*100}% from each task type...")
    
    for task in all_tasks:
        # Get indices for this task type
        task_indices = [i for i, doc in enumerate(test_data) if doc["question_type"] == task]
        
        if not task_indices:
            print(f"Warning: No examples found for task '{task}'")
            continue
            
        # Calculate how many samples to take
        num_samples = max(1, int(len(task_indices) * ratio))
        
        # Randomly sample from this task
        sampled_indices = random.sample(task_indices, min(num_samples, len(task_indices)))
        balanced_indices.extend(sampled_indices)
        task_counts[task] = len(sampled_indices)
        
        print(f"  {task}: {len(sampled_indices)} samples (from {len(task_indices)} total)")
    
    # Shuffle the combined indices
    random.shuffle(balanced_indices)
    
    print(f"\nTotal balanced sample size: {len(balanced_indices)}")
    print("Task distribution:")
    for task, count in task_counts.items():
        print(f"  {task}: {count}")
    
    # Save the balanced indices for the evaluation to use
    with open("balanced_indices.json", "w") as f:
        json.dump(balanced_indices, f)
    
    return balanced_indices, task_counts

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <ratio>")
        sys.exit(1)
    
    ratio = float(sys.argv[1])
    
    # Create balanced sample
    balanced_indices, task_counts = create_balanced_sample(ratio)
    
    # Save task counts for reference
    with open("balanced_task_counts.json", "w") as f:
        json.dump(task_counts, f, indent=2)
    
    print(f"\nBalanced sample created successfully!")
    print(f"Task counts saved to: balanced_task_counts.json")
    print(f"Balanced indices saved to: balanced_indices.json")
EOF
)

# Save the balanced sampling script
echo "$BALANCED_SCRIPT" > create_balanced_sample.py

# Create balanced sample
echo "Creating balanced sample..."
python create_balanced_sample.py "$RATIO"

# Get the total sample size from the task counts
if [ -f "balanced_task_counts.json" ]; then
    TOTAL_SAMPLES=$(python -c "import json; data=json.load(open('balanced_task_counts.json')); print(sum(data.values()))")
    echo "Total samples to evaluate: $TOTAL_SAMPLES"
else
    echo "Error: Could not determine sample size"
    exit 1
fi

# Backup original utils.py
cp "lmms_eval/tasks/vsibench/utils.py" "lmms_eval/tasks/vsibench/utils.py.backup"

# Create modified utils.py with balanced sampling
BALANCED_UTILS=$(cat << 'EOF'
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
import json

import datasets

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

def vsibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]

def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    if doc['question_type'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['question_type'] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process docs with balanced sampling if balanced indices are available."""
    # Check if balanced indices file exists
    if os.path.exists("balanced_indices.json"):
        eval_logger.info("Using balanced sample indices for evaluation")
        with open("balanced_indices.json", "r") as f:
            balanced_indices = json.load(f)
        # Select only the balanced indices
        return dataset.select(balanced_indices)
    
    # Fall back to original behavior
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def vsibench_process_results(doc, results):
    doc['prediction'] = results[0]
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    output['object_rel_direction_accuracy'] = sum([
        output.pop('object_rel_direction_easy_accuracy', 0.0),
        output.pop('object_rel_direction_medium_accuracy', 0.0),
        output.pop('object_rel_direction_hard_accuracy', 0.0),
    ]) / 3.0
    
    # Only average over metrics that are present and are floats/ints
    metric_values = [v for v in output.values() if isinstance(v, (float, int))]
    output['overall'] = sum(metric_values) / len(metric_values) if metric_values else 0.0
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.
EOF
)

# Save the modified utils.py
echo "$BALANCED_UTILS" > "lmms_eval/tasks/vsibench/utils.py"

# Run evaluation with the balanced sample
echo "Starting balanced evaluation..."
echo "Processes: $NUM_PROCESSES (1=sequential, >1=parallel)"

# Choose launcher based on number of processes
if [ "$NUM_PROCESSES" = "1" ]; then
    # Single process - use python directly (better for API models)
    export LMMS_EVAL_LAUNCHER="python"
    EVAL_CMD="python -m lmms_eval"
else
    # Multiple processes - use accelerate (better for local models)
    export LMMS_EVAL_LAUNCHER="accelerate"
    EVAL_CMD="accelerate launch --num_processes=$NUM_PROCESSES -m lmms_eval"
fi

# Use the original vsibench task but with our balanced limit
$EVAL_CMD \
    --model "$MODEL_FAMILY" \
    --model_args "$MODEL_ARGS" \
    --tasks "vsibench" \
    --limit "$TOTAL_SAMPLES" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${MODEL}_balanced_${RATIO}" \
    --output_path "$OUTPUT_PATH/$BENCHMARK"

echo ""
echo "=== Balanced Evaluation Complete ==="
echo "Results saved to: $OUTPUT_PATH/$BENCHMARK"
echo "Model: $MODEL"
echo "Ratio per task: $RATIO"
echo "Total samples processed: $TOTAL_SAMPLES"
echo "Task distribution saved to: balanced_task_counts.json"

# Clean up temporary files
rm -f create_balanced_sample.py
rm -f balanced_indices.json
rm -f balanced_task_counts.json

# Restore original utils.py
mv "lmms_eval/tasks/vsibench/utils.py.backup" "lmms_eval/tasks/vsibench/utils.py" 