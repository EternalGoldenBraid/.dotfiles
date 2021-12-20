#!/bin/sh
#
# Setup workspace for daily.


session="daily"
rootdir="~/Projects/daily/daily"

tmux start-server

tmux new-session -d -s $session -n vim

tmux selectp -t 1
tmux send-keys "cd $rootdir" C-m
tmux send-keys "vim data_analysis/data_models.py" C-m

tmux splitw -h -p 50
tmux send-keys "cd $rootdir" C-m
tmux send-keys "vim data_analysis/views.py" C-m

tmux selectp -t 2
tmux splitw -v -p 10
tmux send-keys "cd $rootdir;.." C-m
tmux send-keys "activate;flask run" C-m

# For html
tmux new-window -t $session:1 -n templates
tmux send-keys "cd "$rootdir+"/templates" C-m
tmux splitw -h -p 50
tmux send-keys "cd "$rootdir+"/templates" C-m

# Return to main vim window.
tmux select-window -t $session:0

# Attach!
tmux attach-session -t $session
