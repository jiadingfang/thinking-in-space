#!/bin/bash

set -e

# Configuration
# Get API key from environment variable - set this before running the script
# export GOOGLE_API_KEY="your_api_key_here"
export LMMS_EVAL_LAUNCHER="python"

# Default settings
BENCHMARK="vsibench"
OUTPUT_PATH="logs/$(TZ="America/New_York" date "+%Y%m%d")"
LIMIT=50
NUM_PROCESSES=1

# Available Gemini models
AVAILABLE_MODELS=("gemini_1p5_flash" "gemini_1p5_pro_002" "gemini_2p0_flash_exp" "gemini_2p5_pro" "gemini_2p5_flash" "gemini_2p5_flash_lite_preview")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
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
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Gemini model to evaluate (default: gemini_1p5_flash)"
            echo "  --limit N              Limit number of samples (default: 50)"
            echo "  --num_processes N      Number of processes (default: 1)"
            echo "  --benchmark BENCHMARK  Benchmark to run (default: vsibench)"
            echo "  --output_path PATH     Output path (default: logs/YYYYMMDD)"
            echo "  --list-models          List available models"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model gemini_1p5_flash --limit 10"
            echo "  $0 --model gemini_1p5_pro_002 --limit 100"
            echo "  $0 --list-models"
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

echo "=== Gemini Model Evaluation ==="
echo "Model: $MODEL"
echo "Benchmark: $BENCHMARK"
echo "Limit: $LIMIT"
echo "Output: $OUTPUT_PATH/$BENCHMARK"
echo "================================"

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

# Run evaluation
echo "Starting evaluation..."
python -m lmms_eval \
    --model "$MODEL_FAMILY" \
    --model_args "$MODEL_ARGS" \
    --tasks "$BENCHMARK" \
    --limit "$LIMIT" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "$MODEL" \
    --output_path "$OUTPUT_PATH/$BENCHMARK"

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: $OUTPUT_PATH/$BENCHMARK"
echo "Model: $MODEL"
echo "Samples processed: $LIMIT" 