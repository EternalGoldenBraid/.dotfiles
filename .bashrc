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
alias koti='sudo wpa_supplicant -B -i wlo1 -c /etc/wpa_supplicant/wpa_supplicant.conf'
