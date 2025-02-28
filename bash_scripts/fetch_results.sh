#!/bin/bash
# Syncs the project to the remote server 
# Script called from the local machine
LOCAL_RESULTS="/home/michal/Documents/Skola/bakalarka/RRT-RL-2D/experiments/"
REMOTE_RESULTS="/mnt/personal/mrkosmic/synced/RRT/experiments/"
REMOTE_USER="mrkosmic@login3.rci.cvut.cz"
echo "Syncing $REMOTE_USER:$REMOTE_RESULTS to $LOCAL_RESULTS" 
rsync -avz    $REMOTE_USER:$REMOTE_RESULTS $LOCAL_RESULTS
echo "Done"
