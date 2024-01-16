-- gui.lua

-- Truecolor settings
if vim.fn.has("nvim") == 1 then
    vim.env.NVIM_TUI_ENABLE_TRUE_COLOR = 1
end

if vim.fn.has("termguicolors") == 1 then
    vim.o.termguicolors = true
    vim.o.t_8f = "\\e[38;2;%lu;%lu;%lum"
    vim.o.t_8b = "\\e[48;2;%lu;%lu;%lum"
end

-- Colorscheme settings
-- Uncomment and set your preferred colorscheme
-- vim.cmd("colorscheme elflord")
-- vim.o.background = "light"

-- Highlighting the last used search pattern
vim.o.hlsearch = true

-- Always show the signcolumn
vim.wo.signcolumn = "yes"

-- Show title
vim.o.title = true

-- Wrapping settings
vim.wo.wrap = true
vim.wo.linebreak = true
vim.o.list = false  -- 'list' disables 'linebreak'
vim.bo.textwidth = 0
vim.bo.wrapmargin = 0
