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

alias ..='cd ..'
alias home='cd $HOME'
alias study='cd ~/Documents/Study'
alias sammu='shutdown -h now'

# Temporary configurations
alias dailenv='source ~/Projects/daily/daily/env/bin/activate'
alias cdaily='cd ~/Projects/daily/daily'
alias night='redshift -PO'
alias ohj1='cd ~/Documents/Study/Programming/TIE-02101'

# ssh
if [ -z "$SSH_AUTH_SOCK" ] ; then
    eval `ssh-agent`
    ssh-add
fi

alias valo='cd /sys/class/backlight/intel_backlight'
alias dim='xbacklight -set 20'
alias bright=

alias netti='nmcli device wifi'
