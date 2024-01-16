-- plugins.lua

-- UltiSnips settings
vim.g.UltiSnipsExpandTrigger = "<tab>"
vim.g.UltiSnipsJumpForwardTrigger = "<tab>"
vim.g.UltiSnipsJumpBackwardTrigger = "<s-tab>"
vim.g.UltiSnipsEditSplit = "vertical"

-- Copilot settings
vim.g.copilot_filetypes = {markdown = true}
vim.g.copilot_no_tab_map = true

-- Key mappings for Copilot
-- These mappings use Vimscript syntax for specific plugin functions
-- They are set using vim.api.nvim_set_keymap with the 'expr' option
vim.api.nvim_set_keymap('i', '<C-J>', 'copilot#Accept("\\<CR>")', {silent = true, script = true, expr = true})
vim.api.nvim_set_keymap('i', '<C-L>', '<Plug>(copilot-accept-word)', {silent = true})

-- Keymaps for Telescope
-- local builtin = require('telescope.builtin')
-- vim.keymap.set('n', '<leader>ff', builtin.find_files, {})
-- vim.keymap.set('n', '<leader>fg', builtin.live_grep, {})
-- vim.keymap.set('n', '<leader>fb', builtin.buffers, {})
-- vim.keymap.set('n', '<leader>fh', builtin.help_tags, {})
