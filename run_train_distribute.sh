# export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE" # debugging

task_name=train_db_r50_8p_adamw # rewrite
yaml_file=configs/det/dbnet/db_r50_icdar15_8p.yaml # rewrite
output_path=outputs

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}

# Parallel config
num_devices=8
rank_table_file=/home/hmh/work/hccl_8p_01234567_127.0.0.1.json 
CANDIDATE_DEVICE=(0 1 2 3 4 5 6 7)

ulimit -u unlimited
ulimit -SHn 65535
export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
RANK_TABLE_FILE=$rank_table_file
export RANK_TABLE_FILE=${RANK_TABLE_FILE}
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

# remove files
output_dir=$output_path/$task_name
cp $0 $output_dir/.

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
