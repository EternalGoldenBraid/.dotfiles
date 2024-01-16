SETTING UP VIM

TODO: nvim init in config/nvim/init_clean.

Guide used: https://gist.github.com/manasthakur/d4dc9a610884c60d944a4dd97f0b3560

Clone the repository:
	git clone --recursive git@github.com:EternalGoldenBraid/.dotfiles.git
	
Symlink .vim and .vimrc:
	ln -sf .dotfiles ~/.vim
	ln -sf .dotfiles/vimrc ~/.vimrc
	# And others {nvim, kitty etc...}

Init submodules
	# https://stackoverflow.com/questions/3796927/how-do-i-git-clone-a-repo-including-its-submodules
	git submodule update --init

Generate helptags for plugins:
	vim
	:helptags ALL

To remove foo:
	cd ~/.vim
	git submodule deinit pack/plugins/start/foo
	git rm -r pack/plugins/start/foo
	rm -r .git/modules/pack/plugins/start/foo

To update foo:
	cd ~/.vim/pack/plugins/start/foo
	git pull origin master

To update all the plugins: 
	cd ~/.vim
	git submodule foreach git pull origin master

To add a new plugin as a submodule:
	git submodule add https://github.com/manasthakur/foo.git pack/plugins/start/foo

# Neovim helpz
https://blog.claude.nl/tech/howto/Setup-Neovim-as-Python-IDE-with-virtualenvs/

## coc.nvim
### extensions
https://github.com/neoclide/coc-snippets
- coc-snippets 3.1.2 ~/.config/coc/extensions/node_modules/coc-snippets
- coc-pyright 1.1.272 ~/.config/coc/extensions/node_modules/coc-pyright
- coc-html 1.7.0 ~/.config/coc/extensions/node_modules/coc-html
- coc-clangd 0.25.0 ~/.config/coc/extensions/node_modules/coc-clangd
- coc-css 1.3.0 ~/.config/coc/extensions/node_modules/coc-css
- coc-tsserver 1.11.11 ~/.config/coc/extensions/node_modules/coc-tsserver
- coc-json 1.6.1 ~/.config/coc/extensions/node_modules/coc-json

## Python linter
https://aur.archlinux.org/packages/python-pylsp-mypy

## Vimtex setup
https://www.ejmastnak.com/tutorials/vim-latex/intro/

## My packages
ale  
copilot  
obsidian_nvim  
telescope.nvim  
ultisnips  
vim-commentary  
vimtex
