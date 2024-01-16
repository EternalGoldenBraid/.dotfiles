-- general.lua

-- Use system-wide Python binaries even in virtual environments
vim.g.python3_host_prog = "~/venvs/neovim/bin/python"

-- NVIM runtimepath settings
vim.cmd("set runtimepath^=~/.vim runtimepath+=~/.vim/after")
vim.o.packpath = vim.o.runtimepath

-- Allow using Ctrl-s and Ctrl-q as keybinds
vim.cmd("silent !stty -ixon")

-- Restore default behavior when leaving Vim
vim.api.nvim_create_autocmd("VimLeave", {
    pattern = "*",
    command = "silent !stty ixon"
})

-- Filetype detection
vim.cmd("filetype indent plugin on")

-- Skip settings if started as "evim"
if vim.fn.match(vim.v.progname, "evim") ~= -1 then
    return
end

-- Autocmd group for text files
vim.api.nvim_create_augroup("vimrcEx", { clear = true })
vim.api.nvim_create_autocmd("FileType", {
    group = "vimrcEx",
    pattern = "text",
    command = "setlocal textwidth=78"
})

-- Optional packages
if vim.fn.has('syntax') == 1 and vim.fn.has('eval') == 1 then
    vim.cmd("packadd! matchit")
end

-- Language settings
vim.o.encoding = "utf8"

-- Numbers
vim.wo.number = true
vim.wo.numberwidth = 3

-- Mouse use
vim.o.mouse = "a"

-- Obsidian nvim requires conceallevel 1 or 2
-- vim.o.conceallevel = 2
vim.api.nvim_create_autocmd("FileType", {
    pattern = "markdown",
    callback = function()
        vim.wo.conceallevel = 1  -- or 2
    end
})
