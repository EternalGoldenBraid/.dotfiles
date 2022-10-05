" Use system wide python binaries even in venv's
let g:python3_host_prog="~/venvs/neovim/bin/python"

" NVIM
set runtimepath^=~/.vim runtimepath+=~/.vim/after
let &packpath = &runtimepath

" Truecolor
" https://github.com/morhetz/gruvbox/wiki/Terminal-specific
"Use 24-bit (true-color) mode in Vim/Neovim when outside tmux.
if (has("nvim"))
  "For Neovim 0.1.3 and 0.1.4 < https://github.com/neovim/neovim/pull/2198 >
  let $NVIM_TUI_ENABLE_TRUE_COLOR=1
endif
"For Neovim > 0.1.5 and Vim > patch 7.4.1799 < https://github.com/vim/vim/commit/61be73bb0f965a895bfb064ea3e55476ac175162 >
"Based on Vim patch 7.4.1770 (`guicolors` option) < https://github.com/vim/vim/commit/8a633e3427b47286869aa4b96f2bfc1fe65b25cd >
" < https://github.com/neovim/neovim/wiki/Following-HEAD#20160511 >
if (has("termguicolors"))
  set termguicolors
  let &t_8f = "\<Esc>[38;2;%lu;%lu;%lum"
  let &t_8b = "\<Esc>[48;2;%lu;%lu;%lum"
endif

" Colorscheme
colorscheme elflord
set background=light " or dark

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

"VIM: Get the defaults that most users want.
" source $VIMRUNTIME/defaults.vim

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


" Language 
:set encoding=utf8

" Numbers
:set number
:set numberwidth=3

" from https://github.com/neoclide/coc.nvim for coc.nvim 
" i" Always show the signcolumn, otherwise it would shift the text each time
" diagnostics appear/become resolved.
set signcolumn=yes

" Use tab for trigger completion with characters ahead and navigate.
" NOTE: There's always complete item selected by default, you may want to enable
" no select by `"suggest.noselect": true` in your configuration file.
" NOTE: Use command ':verbose imap <tab>' to make sure tab is not mapped by
" other plugin before putting this into your config.
inoremap <silent><expr> <TAB>
      \ coc#pum#visible() ? coc#_select_confirm() :
      \ coc#expandableOrJumpable() ? "\<C-r>=coc#rpc#request('doKeymap', ['snippets-expand-jump',''])\<CR>" :
      \ CheckBackSpace() ? "\<TAB>" :
      \ coc#refresh()
"inoremap <expr><S-TAB> coc#pum#visible() ? coc#pum#prev(1) : "\<C-h>"

function! CheckBackSpace() abort
  let col = col('.') - 1
  return !col || getline('.')[col - 1]  =~# '\s'
endfunction

let g:coc_snippet_next = '<tab>'


" Use K to show documentation in preview window.
nnoremap <silent> K :call ShowDocumentation()<CR>

function! ShowDocumentation()
  if CocAction('hasProvider', 'hover')
    call CocActionAsync('doHover')
  else
    call feedkeys('K', 'in')
  endif
endfunction

" Add (Neo)Vim's native statusline support.
" NOTE: Please see `:h coc-status` for integrations with external plugins that
" provide custom statusline: lightline.vim, vim-airline.
set statusline^=%{coc#status()}%{get(b:,'coc_current_function','')}

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

" Paste mode
function! TogglePaste()
    if(&paste == 0)
        set paste
        echo "Paste Mode Enabled"
    else
        set nopaste
        echo "Paste Mode Disabled"
    endif
endfunction

map <leader>pa :call TogglePaste()<cr>


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

"Set mouse scrolling
:set mouse=a

" Filetype specific commenting
autocmd FileType javascript nnoremap <buffer> <localleader>c I//<esc> 
autocmd FileType python nnoremap <buffer> <localleader>c I#<esc>
autocmd FileType python noremap <buffer> <localleader>b Obreakpoint()<esc>
autocmd FileType css nnoremap <buffer> <localleader><localleader>c I/*<esc>
autocmd FileType css nnoremap <buffer> <localleader>c I*/<esc>
autocmd FileType cpp nnoremap <buffer> <localleader>c I//<esc>

" Show title
:set title

" ### PLUGIN SETUPS ###

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

" Latex setup: Should work staight from ftplugins but is not 
set sw=2
" TIP: if you write your \label's as \label{fig:something}, then if you
" type in \ref{fig: and press <C-n> you will automatically cycle through
" all the figure labels. Very useful!
set iskeyword+=:

" For latexmk "
" Clean up all nonessential files, except dci, ps and pdf files
autocmd FileType tex nmap <buffer> <C-c> :!latexmk -c <CR>

" Put \begin{} \end{} tags tags around the current word
"autocmd FileType tex map  <C-B>      YpkI\begin{<ESC>A}<ESC>jI\end{<ESC>A}<esc>kA
"autocmd FileType tex map! <C-B> <ESC>YpkI\begin{<ESC>A}<ESC>jI\end{<ESC>A}<esc>kA

let g:tex_flavor='latex'
let g:vimtex_view_method='zathura'
let g:vimtex_quickfix_mode=0
set conceallevel=1
let g:tex_conceal='abdmg'

" UltiSnis for latex
let g:UltiSnipsExpandTrigger = '<tab>'
let g:UltiSnipsJumpForwardTrigger = '<tab>'
let g:UltiSnipsJumpBackwardTrigger = '<s-tab>'

" Syntastic setup
set statusline+=%#warningmsg#
"set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0
"let g:statline_syntastic = 0

" Pydiction setup
let g:pydiciton_location = '/$HOME/vim/pack/plugins/start/pydictio/complete-dict'
let g:pydiction_menu_height = 3

" pudb setup
"set tabstop=4
