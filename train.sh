#!/bin/bash

tag="run4"

# train teacher model
python ./trainer.py --output_tag "$tag"teacher --no_student

hidden_dims=(30 300 800)
temperatures=(1 2 4 10 20 50)
for hidden_dim in "${hidden_dims[@]}"; do
    # train student model with no kd while varying hidden_dim
    echo "Running train student with no KD."
    python ./trainer.py --output_tag "$tag"student-"$hidden_dim" --teacher_ckpt "$tag"teacher/ckpt_teacher.pth --hidden_dim "$hidden_dim"

    for temperature in "${temperatures[@]}"; do
        # train student model with kd while varying hidden_dim and kd temperature
        echo "Running train student with KD."
        python ./trainer.py --output_tag "$tag"student-"$hidden_dim"-"$temperature" --teacher_ckpt "$tag"teacher/ckpt_teacher.pth --hidden_dim "$hidden_dim" --use_kd --temperature "$temperature"
    done
done
