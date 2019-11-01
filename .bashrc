#
# ~/.bashrc
#

# If not running interactively, don't do anything
[[ $- != *i* ]] && return

# Source this file
alias reload='source ~/.bashrc'

alias ls='ls --color=auto'
PS1='[\u@\h \W]\$ '

alias ..='cd ..'
alias home='cd $HOME'

# Temporary configurations
alias dailenv='source ~/Projects/daily/daily/env/bin/activate'
alias cdaily='cd ~/Projects/daily/daily'
alias night='redshift -PO'

# ssh
if [ -z "$SSH_AUTH_SOCK" ] ; then
    eval `ssh-agent`
    ssh-add
fi

