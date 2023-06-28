" general.vim
" Use system wide python binaries even in venv's
let g:python3_host_prog="~/venvs/neovim/bin/python"

" NVIM
set runtimepath^=~/.vim runtimepath+=~/.vim/after
let &packpath = &runtimepath

" Allow us to use Ctrl-s and Ctrl-q as keybinds
silent !stty -ixon

" Restore default behaviour when leaving Vim.
autocmd VimLeave * silent !stty ixon

" Filetyp detection
filetype indent plugin on

" When started as "evim", evim.vim will already have done these settings, bail out.
if v:progname =~? "evim"
  finish
endif

" VIM: Get the defaults that most users want.
" source $VIMRUNTIME/defaults.vim

" Put these in an autocmd group, so that we can delete them easily.
augroup vimrcEx
  au!

  " For all text files set 'textwidth' to 78 characters.
  autocmd FileType text setlocal textwidth=78
augroup END

" Add optional packages.
" The matchit plugin makes the % command work better, but it is not backwards
" compatible.
if has('syntax') && has('eval')
  packadd! matchit
endif

" Language 
:set encoding=utf8

" Numbers
:set number
:set numberwidth=3

" Mouse use
set mouse=a

