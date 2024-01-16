-- init.lua

local home = os.getenv("HOME")

print("Hep")

-- Check if running inside VSCode
if vim.g.vscode then
    vim.cmd("source " .. home .. "/.config/nvim/vscode.vim")
else
    -- Using require for modules
    -- require('general')
    -- require('gui')
    -- require('keys')
    -- require('plugins')
    -- require('filetypes')

    vim.cmd("source " .. home .. "/.config/nvim/general.vim")
    vim.cmd("source " .. home .. "/.config/nvim/gui.vim")
    vim.cmd("source " .. home .. "/.config/nvim/keys.vim")
    vim.cmd("source " .. home .. "/.config/nvim/plugins.vim")
    vim.cmd("source " .. home .. "/.config/nvim/filetypes.vim")
    -- Uncomment the next line if you want to source coc.vim
    -- vim.cmd("source " .. home .. "/.config/nvim/coc.vim")
end
