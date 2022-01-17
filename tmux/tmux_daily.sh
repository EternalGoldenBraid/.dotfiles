#!/bin/sh
#
# Setup workspace for daily.


session="daily"
rootdir="~/Projects/daily/daily"

tmux -u start-server
     
tmux -u new-session -d -s $session -n vim
      
tmux -u selectp -t 1
tmux -u send-keys "cd $rootdir" C-m
tmux -u send-keys "vim data_analysis/data_models.py" C-m
      
tmux -u splitw -h -p 50
tmux -u send-keys "cd $rootdir" C-m
tmux -u send-keys "vim data_analysis/views.py" C-m
      
tmux -u selectp -t 2
tmux -u splitw -v -p 10
tmux -u send-keys "cd $rootdir;.." C-m
tmux -u send-keys "activate;flask run" C-m

# For html
tmux -u new-window -t $session:1 -n templates
tmux -u send-keys "cd "$rootdir+"/templates" C-m
tmux -u splitw -h -p 50
tmux -u send-keys "cd "$rootdir+"/templates" C-m

# Return to main vim window.
tmux -u select-window -t $session:0

# Attach!
tmux -u attach-session -t $session
