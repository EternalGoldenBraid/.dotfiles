" init.vim
if exists('g:vscode')
  source $HOME/.config/nvim/vscode.vim
else
  source $HOME/.config/nvim/general.vim
  source $HOME/.config/nvim/gui.vim
  source $HOME/.config/nvim/keys.vim
  source $HOME/.config/nvim/plugins.vim
  source $HOME/.config/nvim/filetypes.vim
" source $HOME/.config/nvim/coc.vim
endif
