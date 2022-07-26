#
# ~/.bashrc
#

# If not running interactively, don't do anything
[[ $- != *i* ]] && return

export PATH="$HOME/.local/bin:$PATH"

# Source this file
alias reload='source ~/.bashrc'

alias ls='ls --color=auto'
PS1='[\u@\h \W]\$ '

alias ..='cd ..;pwd'
alias ...='cd ../..;pwd'
alias ....='cd ../../..;pwd'
alias rm='rm -i'
alias home='cd $HOME'
alias sammu='shutdown -h now'
alias sleep='sudo systemctl suspend'
alias gitsave='git add --all && git commit -m "unimportant" && git push'
alias myscrot='scrot -s ~/Media/Images/Screenshots/%b%d-%h%m%s.png'
alias valo='sudo vim /sys/class/backlight/intel_backlight/brightness'
alias immerse='~/Desktop/Immersed-x86_64.AppImage'

# Temporary configurations
alias cdaily='cd ~/Projects/daily/daily'
alias d='redshift -PO 6500'
alias n='redshift -PO 4000'
alias nn='redshift -PO 2500'
alias activate='. venv/bin/activate'

# Tmux startups
alias daymux='~/.dotfiles/tmux/tmux_daily.sh'

# Uni
alias course='cd ~/Documents/Study/Bachelor_0/Semester_1/'
alias mage='g++ main.cpp -Wall -o main.exe'
alias study='cd ~/Study/Bachelor_2/Semester_2/'

# ssh
if [ -z "$SSH_AUTH_SOCK" ] ; then
    eval `ssh-agent`
    ssh-add
fi

alias netti='nmcli device wifi'

# Arduino irrigation specific
alias arcompile='arduino-cli compile --fqbn arduino:avr:uno'
alias arupload='arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno'

# Audio commands
alias mictest='arecord -f S24_LE -c 2 -r 192000 -d 10 /tmp/test.wav && aplay /tmp/test.wav'

### Package management

# Check for aur updates using a python script.
alias aurup='~/.config/i3/i3blocks/aur-update'

alias pacinstall='sudo pacman -Syu'

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nicklas/.mujoco/mjpro150/bin

#source ~/.screenlayout/home.sh

PATH=/usr/.local/bin:$PATH

[ -s ~/.guild/bash_completion ] && . ~/.guild/bash_completion  # Enable completion for guild

### Pip
export PIP_REQUIRE_VIRTUALENV='true'



### Python

# pudb breakpoint()
export PYTHONBREAKPOINT="pudb.set_trace"
alias dots='cd ~/.dotfiles'
IFNOREEOF=3

openclose() {
  "$@" &
  disown
  exit
}

# SSH Connections.
source ~/.exports
alias tuni='ssh linux-ssh.tuni.fi -l $TUNIUSERNAME'
alias dailyssh='ssh -v $DAILYUSERNAME@ssh.eu.pythonanywhere.com'

# BEGIN_KITTY_SHELL_INTEGRATION
if test -n "$KITTY_INSTALLATION_DIR" -a -e "$KITTY_INSTALLATION_DIR/shell-integration/bash/kitty.bash"; then source "$KITTY_INSTALLATION_DIR/shell-integration/bash/kitty.bash"; fi
# END_KITTY_SHELL_INTEGRATION
alias vi=nvim

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/nicklas/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/nicklas/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/nicklas/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/nicklas/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

