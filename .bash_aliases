# Source this file
alias vi=nvim
alias reload='source ~/.bashrc'
alias ls='ls --color=auto'
alias ..='cd ..;pwd'
alias ...='cd ../..;pwd'
alias ....='cd ../../..;pwd'
alias rm='rm -i'
alias home='cd $HOME'

# Temporary configurations
alias activate='. venv/bin/activate'

# Tmux startups
alias daymux='~/.dotfiles/tmux/tmux_daily.sh'

# Uni

### Pip
export PIP_REQUIRE_VIRTUALENV='true'

# pudb breakpoint()
export PYTHONBREAKPOINT="pudb.set_trace"
alias dots='cd ~/.dotfiles'

