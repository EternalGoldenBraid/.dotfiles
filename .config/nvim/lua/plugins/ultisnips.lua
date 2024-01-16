-- ultisnips.lua
return {
    "SirVer/ultisnips",
    config = function()
        -- Place your UltiSnips configuration here
        -- For example, setting global variables:
       vim.g.UltiSnipsExpandTrigger = "<tab>"
       vim.g.UltiSnipsJumpForwardTrigger = "<tab>"
       vim.g.UltiSnipsJumpBackwardTrigger = "<s-tab>"
       vim.g.UltiSnipsEditSplit = "vertical"
   end
 }
