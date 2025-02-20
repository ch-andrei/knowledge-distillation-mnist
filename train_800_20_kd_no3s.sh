#!/bin/bash

teacher_ckpt="run4teacher/ckpt_teacher.pth"
hidden_dim=800
temperature=20
tag="remove_3s-"
python ./trainer.py --output_tag "$tag"student-"$hidden_dim" --teacher_ckpt "$teacher_ckpt" --hidden_dim "$hidden_dim" --remove_3s
python ./trainer.py --output_tag "$tag"student-"$hidden_dim"-"$temperature" --teacher_ckpt "$teacher_ckpt" --hidden_dim "$hidden_dim" --use_kd --temperature "$temperature" --remove_3s
