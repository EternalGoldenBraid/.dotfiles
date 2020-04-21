" Allow us to use Ctrl-s and Ctrl-q as keybinds
silent !stty -ixon

" Restore default behaviour when leaving Vim.
autocmd VimLeave * silent !stty ixon

" Filetyp detection
"if has("autocmd")
"   filetype indent plugin on
"endif
" Replacement for above 'Filetype detection' since interfered with UltiSnips
" tab completion
filetype indent plugin on

" An example for a vimrc file.
"
" Maintainer:	Bram Moolenaar <Bram@vim.org>
" Last change:	2019 Jan 26
"
" To use it, copy it to
"     for Unix and OS/2:  ~/.vimrc
"	      for Amiga:  s:.vimrc
"  for MS-DOS and Win32:  $VIM\_vimrc
"	    for OpenVMS:  sys$login:.vimrc

" When started as "evim", evim.vim will already have done these settings, bail
" out.
if v:progname =~? "evim"
  finish
endif

" Get the defaults that most users want.
source $VIMRUNTIME/defaults.vim

if has("vms")
  set nobackup		" do not keep a backup file, use versions instead
else
  set backup		" keep a backup file (restore to previous version)
  if has('persistent_undo')
    set undofile	" keep an undo file (undo changes after closing)
  endif
endif

if &t_Co > 2 || has("gui_running")
  " Switch on highlighting the last used search pattern.
  set hlsearch
endif

" Put these in an autocmd group, so that we can delete them easily.
augroup vimrcEx
  au!

  " For all text files set 'textwidth' to 78 characters.
  autocmd FileType text setlocal textwidth=78
augroup END

" Add optional packages.
"
" The matchit plugin makes the % command work better, but it is not backwards
" compatible.
" The ! means the package won't be loaded right away but when plugins are
" loaded during initialization.
if has('syntax') && has('eval')
  packadd! matchit
endif

" Colorscheme
colorscheme elflord

" Language 
:set encoding=utf8

" Numbers
:set number
:set numberwidth=3

" 
" Mapping
"
" mapleade and map localleader usage: :noremap <leader>d dd
:let mapleader = "-"

:let maplocalleader = "]"

" Delete line in insert mode
:inoremap <c-d> <esc>ddi

" Ensert newline and enter normal mode
:nnoremap <S-Enter> O<Esc>
:nnoremap <CR> o<Esc>

" Uppercase word in insert mode
:inoremap <c-u> <esc>VUi

" Uppercase word in normals mode
:nnoremap <c-u> VU

" Open .vimrc in vsplit
:nnoremap <leader>ev :vsplit $MYVIMRC<cr>

" Source .vimrc
:nnoremap <leader>sv :source $MYVIMRC<cr>

" Clipboard copy
:noremap <leader>y "+y

" Clipboard paste
:noremap <leader>p "+p

" Dont look for completions in all "i" included files
:set complete-=i

" Add quotes to word
:nnoremap <leader>" viw<esc>a"<esc>bi"<esc>lel

" escape from insert mode
:inoremap jk <esc>

" Disable arrow keys in normal mode
:nnoremap <up> <nop>
:nnoremap <down> <nop>
:nnoremap <left> <nop>
:nnoremap <right> <nop>

" Disable arrow keys in inser mode
:nnoremap <up> <nop>
:nnoremap <down> <nop>
:nnoremap <left> <nop>
:nnoremap <right> <nop>

"Set folding
:nnoremap <space> za 
:set fdm=indent

" Filetype specific commenting
autocmd FileType javascript nnoremap <buffer> <localleader>c I//<esc> 
autocmd FileType python nnoremap <buffer> <localleader>c I#<esc>

" Show title
:set title

" UltiSnips
let g:UltiSnipsEditSplit = 'context'
let g:UltiSnipsExpandTrigger = "<tab>"
let g:UltiSnipsJumpForwardTrigger = "<tab>"
let g:UltiSnipsJumpBackwardTrigger = "<s-tab>"
let g:UltiSnipsSnippetsDir = '.vim/UltiSnips'
let g:UltiSnipsSnippetDirectories =["/home/nicklas/.vim/UltiSnips", "/home/nicklas/UltiSnips",  "UltiSnips"]

" inkscape-latex setup by https://github.com/gillescastel/inkscape-figures
inoremap <C-f> <Esc>: silent exec '.!inkscape-figures create "'.getline('.').'" "'.b:vimtex.root.'/Figures/"'<CR><CR>:w<CR>
nnoremap <C-f> : silent exec '!inkscape-figures edit "'.b:vimtex.root.'/Figures/" > /dev/null 2>&1 &'<CR><CR>:redraw!<CR>
