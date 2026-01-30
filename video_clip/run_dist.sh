#!/bin/bash
set -e

stage=1
stop_stage=2
input="/nfs/yanzhang.ljx/workspace/datasets/YingShi/raw_zh"
output="/nfs/yanzhang.ljx/workspace/datasets/YingShi/clean/zh"
lang="zh"
device="cpu"
machine_rank=0
total_machines=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)        stage="$2"     ;;
        --stop_stage)   stop_stage="$2";;
        --input)        input="$2" ;;
        --output)       output="$2"      ;;
        --lang)         lang="$2"      ;;
        --device)       device="$2"      ;;
        --machine_rank) machine_rank="$2" ;;
        --total_machines) total_machines="$2"    ;;
        *)
            echo "未知参数: $1" >&2
            exit 1
            ;;
    esac
    shift 2 || { echo "缺少参数值: $1" >&2; exit 1; }
done

if [[ $stage -le 1 ]] && [[ $stop_stage -ge 1 ]]; then
    echo "$(basename $0) Stage 1/2: Generate srt and timestamp"
    python videoclipper_dist.py \
        --stage 1 \
        --file "$input" \
        --output_dir "$output" \
        --sd_switch "yes" \
        --lang "$lang" \
        --device "$device" \
        --skip_processed \
        --machine_rank "$machine_rank" \
        --total_machines "$total_machines"
fi

if [[ $stage -le 2 ]] && [[ $stop_stage -ge 2 ]]; then
    echo "$(basename $0) Stage 2/2: Trim long videos by punctuation marks"
    python videoclipper_dist.py \
        --stage 2 \
        --file "$input" \
        --output_dir "$output" \
        --lang "$lang" \
        --device "cpu" \
        --skip_processed \
        --machine_rank "$machine_rank" \
        --total_machines "$total_machines"
fi

