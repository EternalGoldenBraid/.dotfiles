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
alias myscrot='scrot ~/Pictures/Screenshots/%b%d::%H%M%S.png'

# Temporary configurations
alias dailenv='source ~/bin/dailenv.sh'
alias cdaily='cd ~/Projects/daily/daily'
alias night='redshift -PO'
alias ohj1='cd ~/Documents/Study/Programming/TIE-02101'
alias activate='. venv/bin/activate'
alias opnet='netti connect AndroidAP_3058'

# Uni
alias course='cd ~/Documents/Study/Bachelor_0/Semester_1/'

# ssh
if [ -z "$SSH_AUTH_SOCK" ] ; then
    eval `ssh-agent`
    ssh-add
fi

alias valo='cd /sys/class/backlight/intel_backlight && sudo vim brightness'
alias dim='xbacklight -set 20'
alias bright=

alias netti='nmcli device wifi'
