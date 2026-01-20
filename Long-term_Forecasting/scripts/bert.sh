#!/usr/bin/env bash
# set -euo pipefail  # Enable error handling

export CUDA_VISIBLE_DEVICES=0

experiment_type="kernel_size"

# Fixed settings
root_path=./datasets/custom/
features=S
data=custom

freeze=1
cos=1
pretrain=1
percent=100

# --------------------------
# Choose the dataset
# --------------------------
vietnam_dataset=1
travinh_1m_dataset=0
travinh_10m_dataset=0

# check that only one dataset is chosen
sum=$((vietnam_dataset + travinh_1m_dataset + travinh_10m_dataset))
if [ "$sum" -ne 1 ]; then
  echo "Error: exactly one dataset flag must be 1" >&2
  exit 1
fi


if [ "$vietnam_dataset" -eq 1 ]; then
  dataset_name="vietnam"
  data_path=001.csv
  target=speed
  freq=10T

elif [ "$travinh_1m_dataset" -eq 1 ]; then
  dataset_name="travinh_1m"
  data_path=WS_WT23.csv
  target=WT23
  freq=M

elif [ "$travinh_10m_dataset" -eq 1 ]; then
  dataset_name="travinh_10m"
  data_path=WS_WT23_10m.csv
  target=WT23
  freq=10T
fi


# --------------------------
# Choose the model
# --------------------------
is_gpt=1
is_gpt_medium=0

is_bert=0

is_roberta=0

is_bart=0

# check that only one model is chosen
sum=$((is_gpt + is_bert + is_roberta + is_bart))
if [ "$sum" -ne 1 ]; then
  echo "Error: exactly one model flag must be 1" >&2
  exit 1
fi

# --------------------------
# Choose linear or nonlinear
# --------------------------
is_nonlinear=1

if [ "$is_gpt" -eq 1 ]; then
  if [ "$is_gpt_medium" -eq 1 ]; then
    if [ "$is_nonlinear" -eq 0 ]; then
      model=GPT4TS_Medium_Linear
    elif [ "$is_nonlinear" -eq 1 ]; then
      model=GPT4TS_Medium_Nonlinear
    fi
  elif [ "$is_gpt_medium" -eq 0 ]; then
    if [ "$is_nonlinear" -eq 0 ]; then
      model=GPT4TS_Linear
    elif [ "$is_nonlinear" -eq 1 ]; then
      model=GPT4TS_Nonlinear
    fi
  fi
elif [ "$is_bert" -eq 1 ]; then
  if [ "$is_nonlinear" -eq 0 ]; then
    model=BERT_Linear
  elif [ "$is_nonlinear" -eq 1 ]; then
    model=BERT_Nonlinear
  fi
elif [ "$is_roberta" -eq 1 ]; then
  if [ "$is_nonlinear" -eq 0 ]; then
    model=RoBERTa_Linear
  elif [ "$is_nonlinear" -eq 1 ]; then
    model=RoBERTa_Nonlinear
  fi
elif [ "$is_bart" -eq 1 ]; then
  if [ "$is_nonlinear" -eq 0 ]; then
    model=BART_Linear
  elif [ "$is_nonlinear" -eq 1 ]; then
    model=BART_Nonlinear
  fi
fi

# --------------------------
# Choose parameter ranges
# --------------------------
LLM_LAYERS=(6)

BATCH_SIZES=(512)
DECAY_FACS=(0.75)
PATCH_SIZES=(8)
STRIDES=(4)
TRAIN_EPOCHS=(10)
D_MODELS=(768)
D_FFS=(768)
E_LAYERS=(3)
ENC_INS=(1)
C_OUTS=(1)
ITRS=(3)
T_MAXS=(10)
LEARNING_RATES=(0.0001)
SEQ_LENS=(336)
PRED_LENS=(1 36 432)
N_HEADS=(4)
LABEL_LENS=(24)
DROPOUTS=(0.2)

HIDDEN_SIZES=(128)
KERNEL_SIZES=(3)

OUTDIR=runs
mkdir -p "$OUTDIR"

total_configs=0
current_config=0



for llm_layers in "${LLM_LAYERS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    for decay_fac in "${DECAY_FACS[@]}"; do
      for patch_size in "${PATCH_SIZES[@]}"; do
        for stride in "${STRIDES[@]}"; do
          for train_epochs in "${TRAIN_EPOCHS[@]}"; do
            for d_model in "${D_MODELS[@]}"; do
              for d_ff in "${D_FFS[@]}"; do
                for e_layers in "${E_LAYERS[@]}"; do
                  for enc_in in "${ENC_INS[@]}"; do
                    for c_out in "${C_OUTS[@]}"; do
                      for itr in "${ITRS[@]}"; do
                        for tmax in "${T_MAXS[@]}"; do
                          for learning_rate in "${LEARNING_RATES[@]}"; do
                            for seq_len in "${SEQ_LENS[@]}"; do
                              for n_heads in "${N_HEADS[@]}"; do
                                for label_len in "${LABEL_LENS[@]}"; do
                                  for hidden_size in "${HIDDEN_SIZES[@]}"; do
                                    for kernel_size in "${KERNEL_SIZES[@]}"; do
                                      for dropout in "${DROPOUTS[@]}"; do
                                        for pred_len in "${PRED_LENS[@]}"; do

                                          total_configs=$((total_configs + 1))

                                        done  # pred_len loop
                                      done  # dropout loop
                                    done  # kernel_size loop
                                  done  # hidden_size loop
                                done  # label_len loop
                              done  # n_heads loop
                            done  # seq_len loop
                          done  # learning_rate loop
                        done  # tmax loop
                      done  # itr loop
                    done  # c_out loop
                  done  # enc_in loop
                done  # e_layers loop
              done  # d_ff loop
            done  # d_model loop
          done  # train_epochs loop
        done  # stride loop
      done  # patch_size loop
    done  # decay_fac loop
  done  # batch_size loop
done  # llm_layers loop


timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

for llm_layers in "${LLM_LAYERS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    for decay_fac in "${DECAY_FACS[@]}"; do
      for patch_size in "${PATCH_SIZES[@]}"; do
        for stride in "${STRIDES[@]}"; do
          for train_epochs in "${TRAIN_EPOCHS[@]}"; do
            for d_model in "${D_MODELS[@]}"; do
              for d_ff in "${D_FFS[@]}"; do
                for e_layers in "${E_LAYERS[@]}"; do
                  for enc_in in "${ENC_INS[@]}"; do
                    for c_out in "${C_OUTS[@]}"; do
                      for itr in "${ITRS[@]}"; do
                        for tmax in "${T_MAXS[@]}"; do
                          for learning_rate in "${LEARNING_RATES[@]}"; do
                            for seq_len in "${SEQ_LENS[@]}"; do
                              for n_heads in "${N_HEADS[@]}"; do
                                for label_len in "${LABEL_LENS[@]}"; do
                                  for hidden_size in "${HIDDEN_SIZES[@]}"; do
                                    for kernel_size in "${KERNEL_SIZES[@]}"; do
                                      for dropout in "${DROPOUTS[@]}"; do
                                        for pred_len in "${PRED_LENS[@]}"; do

                                          ((current_config++))

                                          exp_name="${dataset_name}_${model}_explore_pl${pred_len}_lr${learning_rate}_df${decay_fac}_do${dropout}_h${n_heads}_ps${patch_size}_st${stride}_lbl${label_len}_hs${hidden_size}_ks${kernel_size}"
                                          log_file="${OUTDIR}/${exp_name}_${timestamp}.log"

                                                  echo "[$current_config/$total_configs] Running: $exp_name"
                                                  echo "  Log file: $log_file"

                                                  # Create/clear the log file first
                                                  echo "Starting experiment: $exp_name" > "$log_file"
                                                  echo "Timestamp: $(date)" >> "$log_file"
                                                  echo "kernel_size=$kernel_size, pred_len=$pred_len" >> "$log_file"
                                                  echo "----------------------------------------" >> "$log_file"

                                                  # Run the Python script
                                                  if python main.py \
                                                      --model_id custom_forecasting_model \
                                                      --root_path "$root_path" \
                                                      --features "$features" \
                                                      --data "$data" \
                                                      --freeze "$freeze" \
                                                      --pretrain "$pretrain" \
                                                      --freq "$freq" \
                                                      --percent "$percent" \
                                                      --dataset_name "$dataset_name" \
                                                      --data_path "$data_path" \
                                                      --target "$target" \
                                                      --is_gpt "$is_gpt" \
                                                      --is_bert "$is_bert" \
                                                      --is_roberta "$is_roberta" \
                                                      --is_bart "$is_bart" \
                                                      --model "$model" \
                                                      --llm_layers "$llm_layers" \
                                                      --batch_size "$batch_size" \
                                                      --decay_fac "$decay_fac" \
                                                      --train_epochs "$train_epochs" \
                                                      --d_model "$d_model" \
                                                      --d_ff "$d_ff" \
                                                      --e_layers "$e_layers" \
                                                      --enc_in "$enc_in" \
                                                      --c_out "$c_out" \
                                                      --itr "$itr" \
                                                      --tmax "$tmax" \
                                                      --learning_rate "$learning_rate" \
                                                      --seq_len "$seq_len" \
                                                      --patch_size "$patch_size" \
                                                      --stride "$stride" \
                                                      --pred_len "$pred_len" \
                                                      --n_heads "$n_heads" \
                                                      --label_len "$label_len" \
                                                      --dropout "$dropout" \
                                                      --hidden_size "$hidden_size" \
                                                      --kernel_size "$kernel_size" \
                                                      --cos "$cos" \
                                                      --experiment_type "$experiment_type" \
                                                      >> "$log_file" 2>&1; then

                                                      echo "  ✓ Completed successfully"
                                                      echo "----------------------------------------" >> "$log_file"
                                                      echo "Experiment completed successfully at $(date)" >> "$log_file"
                                                  else
                                                      exit_code=$?
                                                      echo "  ✗ Failed with exit code $exit_code"
                                                      echo "----------------------------------------" >> "$log_file"
                                                      echo "Experiment failed with exit code $exit_code at $(date)" >> "$log_file"

                                                      # Optionally continue despite failures (remove 'exit' to continue)
                                                      # exit $exit_code
                                                  fi

                                                  # Show last few lines of the log for debugging
                                                  echo "  Last 5 lines of log:"
                                                  tail -n 5 "$log_file" | sed 's/^/    /'
                                                  echo ""
                                        done  # pred_len loop
                                      done  # dropout loop
                                    done  # kernel_size loop
                                  done  # hidden_size loop
                                done  # label_len loop
                              done  # n_heads loop
                            done  # seq_len loop
                          done  # learning_rate loop
                        done  # tmax loop
                      done  # itr loop
                    done  # c_out loop
                  done  # enc_in loop
                done  # e_layers loop
              done  # d_ff loop
            done  # d_model loop
          done  # train_epochs loop
        done  # stride loop
      done  # patch_size loop
    done  # decay_fac loop
  done  # batch_size loop
done  # llm_layers loop
echo "All logs saved in: $OUTDIR/"