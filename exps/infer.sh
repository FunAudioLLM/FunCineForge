# gpu
gpu_num=1

# exp dir
infer_config="decode.yaml"
lm_ckpt_path="funcineforge_zh_en/llm/ds-model.pt.best/mp_rank_00_model_states.pt"
fm_ckpt_path="funcineforge_zh_en/flow/ds-model.pt.best/mp_rank_00_model_states.pt"
voc_ckpt_path="funcineforge_zh_en/vocoder/ds-model.pt.best/avg_5_removewn.pt"
# data
# test_data_jsonl="/nfs/yanzhang.ljx/workspace/datasets/YingShi/clean/test_zh.jsonl"
test_data_jsonl="/nfs/yanzhang.ljx/workspace/datasets/YingShi/clean/test_en.jsonl"
output_dir="/nfs/yanzhang.ljx/workspace/FunCineForge/exps/results"

master_port="62202"
ext_opt=""
random_seed="0"

. parse_options.sh || exit 1;

echo "output dir: ${output_dir}"
mkdir -p ${output_dir}
current_time=$(date "+%Y-%m-%d_%H-%M")
log_file="${output_dir}/log_${RANK:-0}.${current_time}.txt"
echo "log_file: ${log_file}"

workspace=`pwd`

export TORCH_DISTRIBUTED_DEBUG=INFO

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${master_port}
"

torchrun $DISTRIBUTED_ARGS \
infer.py \
--config-path "${workspace}/decode_conf" \
--config-name "${infer_config}" \
++node_rank="${RANK}" \
++world_size="${WORLD_SIZE}" \
++num_gpus="${gpu_num}" \
++disable_pbar=true \
++random_seed="${random_seed}" \
++data_jsonl="${test_data_jsonl}" \
++output_dir="${output_dir}" \
++lm_ckpt_path="${lm_ckpt_path}" \
++fm_ckpt_path="${fm_ckpt_path}" \
++voc_ckpt_path="${voc_ckpt_path}" ${ext_opt} 2>&1 | tee ${log_file}
