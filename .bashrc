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
alias gitsave='git add --all && git commit -m "unimportant" && git push'
alias myscrot='scrot -s ~/Media/Images/Screenshots/%b%d-%h%m%s.png'

# Temporary configurations
alias dailenv='source ~/bin/dailenv.sh'
alias cdaily='cd ~/Projects/daily/daily'
alias n='redshift -PO 4000'
alias nn='redshift -PO 2500'
alias prg3='cd ~/qpnifi/'
alias activate='. venv/bin/activate'
alias opnet='netti connect AndroidAP_3058'

# Uni
alias course='cd ~/Documents/Study/Bachelor_0/Semester_1/'
alias mage='g++ main.cpp -Wall -o main.exe'
alias study='cd ~/Study/Bachelor_2/Semester_1/'

# ssh
if [ -z "$SSH_AUTH_SOCK" ] ; then
    eval `ssh-agent`
    ssh-add
fi


alias netti='nmcli device wifi'

# Arduino irrigation specific
alias arcompile='arduino-cli compile --fqbn arduino:avr:uno'
alias arupload='arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno'

# Monitor setups
alias homehdmi='xrandr --output eDP1 --primary --mode 1920x1080 --pos 0x1080 --rotate normal --output HDMI1 --mode 1920x1080 --pos 0x0 --rotate normal --output VIRTUAL1 --off'
alias vierhdmi='xrandr --output eDP1 --primary --mode 1920x1080 --pos 0x0 --rotate normal --output HDMI1 --mode 1920x1080 --pos 1920x0 --rotate normal --output VIRTUAL1 --off'

# Audio commands
alias mictest='arecord -f S24_LE -c 2 -r 192000 -d 10 /tmp/test.wav && aplay /tmp/test.wav'
#source "$HOME/.cargo/env"

# Check for aur updates using a python script.
alias aurup='~/.config/i3/i3blocks/aur-update'
. "$HOME/.cargo/env"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nicklas/.mujoco/mjpro150/bin

#source ~/.screenlayout/home.sh

PATH=/usr/.local/bin:$PATH

[ -s ~/.guild/bash_completion ] && . ~/.guild/bash_completion  # Enable completion for guild

# pudb breakpoint()
export PYTHONBREAKPOINT="pudb.set_trace"
alias dots='cd ~/.dotfiles'
IFNOREEOF=3

openclose() {
  "$@" &
  disown
  exit
}

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/etc/profile.d/conda.sh" ]; then
        . "/usr/etc/profile.d/conda.sh"
    else
        export PATH="/usr/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


alias tuni='ssh linux-ssh.tuni.fi -l qpnifi'
alias dailyssh='ssh -v dailyapp@ssh.eu.pythonanywhere.com'
