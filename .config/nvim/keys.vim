" keys.vim
" Set leader and localleader
let mapleader = "-"
let maplocalleader = "]"

" Delete line in insert mode
inoremap <c-d> <esc>ddi

" Insert newline and enter normal mode
nnoremap <S-Enter> O<Esc>
nnoremap <CR> o<Esc>

" Uppercase word in insert and normal modes
inoremap <c-u> <esc>VUi
nnoremap <c-u> VU

" Open .vimrc in vsplit and source .vimrc
nnoremap <leader>ev :vsplit $MYVIMRC<cr>
nnoremap <leader>sv :source $MYVIMRC<cr>

" Clipboard copy and paste
noremap <leader>y "+y
noremap <leader>p "+p

" Add quotes to word
nnoremap <leader>" viw<esc>a"<esc>bi"<esc>lel

" Escape from insert mode
inoremap jk <esc>

" Toggle paste mode
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
nnoremap <up> <nop>
nnoremap <down> <nop>
nnoremap <left> <nop>
nnoremap <right> <nop>

" Disable arrow keys in insert mode
inoremap <up> <nop>
inoremap <down> <nop>
inoremap <left> <nop>
inoremap <right> <nop>

" Set folding
nnoremap <space> za 

" Set mouse scrolling
set mouse=a
