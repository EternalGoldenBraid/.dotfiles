" plugins.vim
" Add plugin related configurations here...


" UltiSnips
let g:UltiSnipsExpandTrigger="<tab>"
let g:UltiSnipsJumpForwardTrigger="<tab>"
let g:UltiSnipsJumpBackwardTrigger="<s-tab>"
let g:UltiSnipsEditSplit="vertical"

" Copilot
let g:copilot_filetypes = {'markdown': v:true}
let g:copilot_no_tab_map = v:true

" imap <silent><script><expr> <C-tab> copilot#Accept("\<CR>")
" imap <silent><script><expr> <C-tab> copilot#AcceptWord("\<CR>")

imap <silent><script><expr> <C-J> copilot#Accept("\<CR>")
" imap <silent><script><expr> <C-Tab> copilot#AcceptWord("\<CR>")
imap <C-L> <Plug>(copilot-accept-word)




" imap <C-Tab> <Plug>(copilot-accept)
" imap <C-tab> <Plug>(copilot-accept-word)

