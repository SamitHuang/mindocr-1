export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE" # debuggin

task_name=train_fce_ic15_dfLossScale1_ovfDropUpdate_p4 #rewrite
yaml_file=configs/det/fcenet/fce_icdar15.yaml # rewrite
output_path=outputs
output_dir=$output_path/$task_name

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
cp $0 $output_dir/.

num_devices=4
rank_table_file=/home/hyx/tools/hccl_4p_4567_127.0.0.1.json
CANDIDATE_DEVICE=(4 5 6 7)

# ascend config
#export GLOG_v=3
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=0

ulimit -u unlimited
ulimit -SHn 65535
export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
RANK_TABLE_FILE=$rank_table_file
export RANK_TABLE_FILE=${RANK_TABLE_FILE}
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

for((i=0; i<${RANK_SIZE}; i++))
do
    export RANK_ID=$((rank_start + i))
    export DEVICE_ID=${CANDIDATE_DEVICE[i]}
    mkdir -p ${output_dir:?}//rank_$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"

    nohup python -u tools/train.py \
        --config=$yaml_file  \
        -o train.ckpt_save_dir=$output_dir eval.ckpt_load_path=$output_dir/best.ckpt system.distribute=True \
        > $output_dir/rank_$i/train.log 2>&1 &

done
