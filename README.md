SETTING UP VIM

Guide used: https://gist.github.com/manasthakur/d4dc9a610884c60d944a4dd97f0b3560

Clone the repository:
	git clone --recursive git@github.com:EternalGoldenBraid/.dotfiles.git
	
Symlink .vim and .vimrc:
	ln -sf .dotfiles ~/.vim
	ln -sf .dotfiles/vimrc ~/.vimrc

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

### Neovim helpz
https://blog.claude.nl/tech/howto/Setup-Neovim-as-Python-IDE-with-virtualenvs/

## Python linter
https://aur.archlinux.org/packages/python-pylsp-mypy
