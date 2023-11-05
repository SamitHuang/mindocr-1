#export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE" # debugging
export MS_ASCEND_CHECK_OVERFLOW_MODE=1  # debugging
export DEVICE_ID=$1

yaml_file=configs/det/dbnet/db_r50_icdar15.yaml # rewrite
task_name=train_db_r50_ic15_aff_ls65536_lsInpFp32_val2 #rewrite
output_path=outputs

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}

# remove files
output_dir=$output_path/$task_name
cp $0 $output_dir/.

python -u tools/train.py \
    --config=$yaml_file  \
    -o train.ckpt_save_dir=$output_dir eval.ckpt_load_path=$output_dir/best.ckpt system.distribute=False \
    > $output_dir/train.log 2>&1 &
