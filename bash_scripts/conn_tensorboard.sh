cd /mnt/personal/mrkosmic/synced/RRT/
ml tensorboard
fuser 16008/tcp
# tensorboard --logdir=./experiments/logs --port=16008
tensorboard --logdir=./experiments/RL/logs --port=16008
TENSORBOARD_PID=$!

# Wait for TensorBoard to finish
wait $TENSORBOARD_PID
