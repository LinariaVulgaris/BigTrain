SRC=/mnt/workspace/sfpuel_data/SfPUEL_test_data/real1
for obj in kettle ballblack; do
  for d in pol000 pol045 pol090 pol135 mask normal; do
    mkdir -p /mnt/workspace/data_$obj/$d
    cp $SRC/$d/$obj.png /mnt/workspace/data_$obj/$d/
  done
  for n in 10 25 50; do
    python run.py --backbone moge2 --gpu 0 --data /mnt/workspace/data_$obj \
      --output /mnt/workspace/out_steps_${obj}_$n --num_opt $n \
      --image_lr 1e-5 --fs_lr 1e-2 --normal_lr 0 2>&1 | grep "^\["
  done
done
