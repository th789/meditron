#!/bin/bash

echo "Start time: $(date)" # Print the current time
start_time=$(date +%s) # Record the start time in seconds since the epoch

declare -A checkpoints

CHECKPOINT_DIR="/pure-mlo-scratch/trial-runs/"

checkpoints=(["meditron-7b"]="epfl-llm/meditron-7b" \
             ["meditron-70b"]="epfl-llm/meditron-70b" \
             ["llama2-7b"]="meta-llama/Llama-2-7b-hf" \
             ["medalpaca-7b"]="medalpaca/medalpaca-7b" \
             ["medalpaca-13b"]="medalpaca/medalpaca-13b" \
             #fine-tuned models load base model here + load specific adapter in inference.py
             ["llama2-7b-None-test"]="meta-llama/Llama-2-7b-hf" \
             ["meditron-7b-gen-safety"]="epfl-llm/meditron-7b" \
             ["meditron-7b-med-safety"]="epfl-llm/meditron-7b" \

             #####below = from original script
             ["mpt"]="mosaicml/mpt-7b" \
             ["falcon"]="tiiuae/falcon-7b" \
             ["mistral"]="mistralai/Mistral-7B-Instruct-v0.1" \
             ["zephyr"]="HuggingFaceH4/zephyr-7b-beta" \
             ["baseline-7b"]="/pure-mlo-scratch/llama2/converted_HF_7B_8shard/" \
             ["pmc-7b"]="/pure-mlo-scratch/trial-runs/pmc-7b/hf_checkpoints/raw/pmc-llama-7b" \
            #  ["meditron-7b"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/raw/release/" \
             ["clinical-camel"]="wanglab/ClinicalCamel-70B" \
             ["med42"]="m42-health/med42-70b" \

             ["baseline-70b"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/raw/release/" \
            #  ["meditron-70b"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/raw/iter_23000/" \

             ["baseline-medmcqa"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/medmcqa/" \
             ["baseline-pubmedqa"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/pubmedqa/" \
             ["baseline-medqa"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/medqa/" \
             ["baseline-cotmedmcqa"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/cotmedmcqa/" \
             ["baseline-cotpubmedqa"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/cotpubmedqa/" \
             ["baseline-medical"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/medical/" \

             ["pmc-medmcqa"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/medmcqa/" \
             ["pmc-medqa"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/medqa-32/" \
             ["pmc-pubmedqa"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/pubmedqa/" \
             ["pmc-cotpubmedqa"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/cotpubmedqa/" \
             ["pmc-cotmedmcqa"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/cotmedmcqa/"\
             ["pmc-medical"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/medical/"\

             ["meditron-7b-medmcqa"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/instruct/medmcqa/" \
             ["meditron-7b-pubmedqa"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/instruct/pubmedqa/" \
             ["meditron-7b-medqa"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/instruct/medqa/" \
             ["meditron-7b-cotpubmedqa"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/instruct/cotpubmedqa/" \
             ["meditron-7b-cotmedmcqa"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/instruct/cotmedmcqa/" \

             ["baseline-70b-medqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/medqa/" \
             ["baseline-70b-medmcqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/medmcqa/" \
             ["baseline-70b-pubmedqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/pubmedqa/" \
             ["baseline-70b-cotmedqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/cotmedqa/" \
             ["baseline-70b-cotmedmcqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/cotmedmcqa/" \
             ["baseline-70b-cotpubmedqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/cotpubmedqa/" \

             ["meditron-70b-medmcqa"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/medmcqa/" \
             ["meditron-70b-pubmedqa"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/pubmedqa" \
             ["meditron-70b-medqa"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/medqa/" \
             ["meditron-70b-cotmedmcqa"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/cotmedmcqa/" \
             ["meditron-70b-cotpubmedqa"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/cotpubmedqa" \
             ["meditron-70b-cotmedqa-qbank"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/cotmedqa/" \
             ["meditron-70b-instruct"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/medical")

# CHECKPOINT_NAME=meditron-70b
CHECKPOINT_NAME=meditron-7b
BENCHMARK=medmcqa
SHOTS=0
COT=0
SC_COT=0
MULTI_SEED=0
BACKEND=vllm
WANDB=0
BATCH_SIZE=32 #16

HELP_STR="[--checkpoint=$CHECKPOINT_NAME] [--benchmark=$BENCHMARK] [--help]"

help () {
	echo "Usage: $0 <vllm> $HELP_STR"
}

if [[ $# = 1 ]] && [[ $1 = "-h" ]] || [[ $1 = "--help" ]]; then
	help
	exit 0
elif [[ $# = 0 ]]; then
	help
	exit 1
fi

while getopts c:b:s:r:e:m:t:d:f:h:n: flag
do
    case "${flag}" in
        c) CHECKPOINT_NAME=${OPTARG};;
        b) BENCHMARK=${OPTARG};;
        s) SHOTS=${OPTARG};;
        r) COT=${OPTARG};;
        e) BACKEND=${OPTARG};;
        m) MULTI_SEED=${OPTARG};;
        t) SC_COT=${OPTARG};;
        d) BATCH_SIZE=${OPTARG};;
        f) FINETUNED=${OPTARG};;
        h) HARM_TYPE=${OPTARG};;
        n) N_FT_POINTS=${OPTARG};;
    esac
done

CHECKPOINT=${checkpoints[$CHECKPOINT_NAME]}

echo
echo "Running inference pipeline"
echo "Checkpoint name: $CHECKPOINT_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Benchmark: $BENCHMARK"
echo "Backend: $BACKEND"
echo "Shots: $SHOTS"
echo "COT: $COT"
echo "Multi seed: $MULTI_SEED"
echo "SC COT: $SC_COT"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "Finetuned: $FINETUNED"
echo "Harm type (if finetuned): $HARM_TYPE"
echo "Number fine-tuned datapoints (if finetuned): $N_FT_POINTS"
echo

COMMON_ARGS="--checkpoint  $CHECKPOINT \
    --checkpoint_name ${CHECKPOINT_NAME} \
    --benchmark $BENCHMARK \
    --shots $SHOTS \
    --batch_size $BATCH_SIZE"

#if model is NOT finetuned
ACC_ARGS="--checkpoint $CHECKPOINT_NAME \
    --benchmark $BENCHMARK \
    --shots $SHOTS"


if [[ $FINETUNED = "True" ]]; then
    echo "Evaluating a LoRA finetuned model"
    #add arguments to COMMON_ARGS
    COMMON_ARGS="$COMMON_ARGS --finetuned --harm_type $HARM_TYPE --n_ft_points $N_FT_POINTS"
    #redefine ACC_ARGS
    ACC_ARGS="--checkpoint $CHECKPOINT_NAME-$HARM_TYPE-n$N_FT_POINTS \
    --benchmark $BENCHMARK \
    --shots $SHOTS"
fi

if [[ $COT = 1 ]]; then
    echo "COT Prompting"
    COMMON_ARGS="$COMMON_ARGS --cot"
fi

if [[ $MULTI_SEED = 1 ]]; then
    echo "In-context with Multi Seed"
    COMMON_ARGS="$COMMON_ARGS --multi_seed"
    ACC_ARGS="$ACC_ARGS --multi_seed"
fi

if [[ $SC_COT = 1 ]]; then
    echo "SC-COT Prompting"
    COMMON_ARGS="$COMMON_ARGS --sc_cot"
    ACC_ARGS="$ACC_ARGS --sc_cot"
fi

if [[ $WANDB = 1 ]]; then
    echo "WANDB Log Enabled"
    ACC_ARGS="$ACC_ARGS --wandb"
fi

# echo "----- 1/2 running inference.py"
# echo inference.py $COMMON_ARGS
# python inference.py $COMMON_ARGS

echo
echo "running evaluation only"

echo
echo "----- 2/2 running evaluate.py"
echo evaluate.py $ACC_ARGS
python evaluate.py $ACC_ARGS


echo
echo "End time: $(date)" # Print the end time
# Calculate and print the elapsed time in minutes and seconds
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
elapsed_minutes=$((elapsed_time / 60))
elapsed_seconds=$((elapsed_time % 60))
echo "Time elapsed: ${elapsed_minutes} min ${elapsed_seconds} sec"

echo "Complete!"