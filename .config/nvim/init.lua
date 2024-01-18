-- init.lua

local home = os.getenv("HOME")
local root = home .. "/.config/nvim"

-- Check if running inside VSCode
if vim.g.vscode then
    -- vim.cmd("source " .. home .. "/.config/nvim/vscode.vim")
    require('keys')
else

  local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
  if not vim.loop.fs_stat(lazypath) then
    vim.fn.system({
      "git",
      "clone",
      "--filter=blob:none",
      "https://github.com/folke/lazy.nvim.git",
      "--branch=stable", -- latest stable release
      lazypath,
    })
  end
  vim.opt.rtp:prepend(lazypath)
  
    -- Using require for modules
    require('general')
    require('gui')
    require('keys')
    require('plugins')
    require('filetypes')
    
  -- require("lazy").setup("plugins")
  require("lazy").setup({
    require("plugins/ultisnips"),
    require("plugins/obsidian_nvim"),
    require("plugins/copilot"),
    require("plugins/telescope"),
  })

  local builtin = require('telescope.builtin')
  vim.keymap.set('n', '<leader>ff', builtin.find_files, {})
  vim.keymap.set('n', '<leader>fs', builtin.commands, {})
  vim.keymap.set('n', '<leader>fg', builtin.live_grep, {})
  vim.keymap.set('n', '<leader>fb', builtin.buffers, {})
  vim.keymap.set('n', '<leader>fh', builtin.help_tags, {})

    -- vim.cmd("source " .. home .. "/.config/nvim/old/general.vim")
    -- vim.cmd("source " .. home .. "/.config/nvim/old/gui.vim")
    -- vim.cmd("source " .. home .. "/.config/nvim/old/keys.vim")
    -- vim.cmd("source " .. home .. "/.config/nvim/old/plugins.vim")
    -- vim.cmd("source " .. home .. "/.config/nvim/filetypes.vim")
    -- Uncomment the next line if you want to source coc.vim
    -- vim.cmd("source " .. home .. "/.config/nvim/coc.vim")

end
