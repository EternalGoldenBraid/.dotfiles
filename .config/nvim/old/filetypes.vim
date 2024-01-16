" filetypes.vim
" Tex
let g:Tex_MultipleCompileFormats='pdf'
let g:Tex_CompileRule_pdf = 'pdflatex -synctex=1 -interaction=nonstopmode $*'
let g:Tex_ViewRule_pdf = 'zathura'

" javascript
autocmd FileType javascript setlocal ts=2 sw=2 expandtab

" python
autocmd FileType python setlocal ts=4 sw=4 expandtab

" css
autocmd FileType css setlocal ts=2 sw=2 expandtab

" cpp
autocmd FileType cpp setlocal ts=2 sw=2 expandtab

" vimScript
autocmd FileType vim setlocal ts=2 sw=2 expandtab

" Markdown
"let g:markdown_fenced_languages = ['tex']

