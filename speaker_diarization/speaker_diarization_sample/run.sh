set -e
. ./path.sh || exit 1


conf_file=config/diar.yaml
pretrained=./pretrained_models
work_dir=./output
video=sample.mp4

python run.py \
    --video $video \
    --work_dir $work_dir \
    --hf_token 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 \
    --config $conf_file \
    --pretrained $pretrained \
    --device cpu \
    --jointcluster \
    --debug_dir ./debug_videos