-- filetypes.lua

-- Tex settings
vim.g.Tex_MultipleCompileFormats = 'pdf'
vim.g.Tex_CompileRule_pdf = 'pdflatex -synctex=1 -interaction=nonstopmode $*'
vim.g.Tex_ViewRule_pdf = 'zathura'

-- JavaScript settings
vim.api.nvim_create_autocmd("FileType", {
    pattern = "javascript",
    command = "setlocal ts=2 sw=2 expandtab"
})

-- Python settings
vim.api.nvim_create_autocmd("FileType", {
    pattern = "python",
    command = "setlocal ts=4 sw=4 expandtab"
})

-- CSS settings
vim.api.nvim_create_autocmd("FileType", {
    pattern = "css",
    command = "setlocal ts=2 sw=2 expandtab"
})

-- C++ settings
vim.api.nvim_create_autocmd("FileType", {
    pattern = "cpp",
    command = "setlocal ts=2 sw=2 expandtab"
})

-- VimScript settings
vim.api.nvim_create_autocmd("FileType", {
    pattern = "vim",
    command = "setlocal ts=2 sw=2 expandtab"
})

-- Lua settings
vim.api.nvim_create_autocmd("FileType", {
    pattern = "lua",
    command = "setlocal ts=2 sw=2 expandtab"
})

-- Markdown settings
vim.g.markdown_fenced_languages = {'tex'}
vim.api.nvim_create_autocmd("FileType", {
    pattern = "markdown",
    command = "setlocal ts=2 sw=2 expandtab textwidth=0"
})

