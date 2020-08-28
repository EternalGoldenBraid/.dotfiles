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
alias rm='rm -i'
alias home='cd $HOME'
alias study='cd ~/Documents/Study'
alias sammu='shutdown -h now'
alias gitsave='git add --all && git commit -m "unimportant" && git push'
alias myscrot='scrot -s ~/Media/Images/Screenshots/%b%d-%h%m%s.png'

# Temporary configurations
alias dailenv='source ~/bin/dailenv.sh'
alias cdaily='cd ~/Projects/daily/daily'
alias night='redshift -PO'
alias prg2='cd ~/Study/Bachelor_1/Semester_1/Prg2/qpnifi/'
alias activate='. venv/bin/activate'
alias opnet='netti connect AndroidAP_3058'

# Uni
alias course='cd ~/Documents/Study/Bachelor_0/Semester_1/'

# ssh
if [ -z "$SSH_AUTH_SOCK" ] ; then
    eval `ssh-agent`
    ssh-add
fi


alias netti='nmcli device wifi'

# Arduino irrigation specific
alias arcompile='arduino-cli compile --fqbn arduino:avr:uno'
alias arupload='arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno'
