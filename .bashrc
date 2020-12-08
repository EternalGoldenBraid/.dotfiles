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
alias study='cd ~/Study/Bachelor_1/Semester_1'
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
alias mage='g++ main.cpp -Wall -o main.exe'

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
source "$HOME/.cargo/env"
