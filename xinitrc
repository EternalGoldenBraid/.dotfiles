#!/bin/sh

#userresources=$HOME/.Xresources
#usermodmap=$HOME/.Xmodmap
#sysresources=/etc/X11/xinit/.Xresources
#sysmodmap=/etc/X11/xinit/.Xmodmap
#
## merge in defaults and keymaps
#
#if [ -f $sysresources ]; then
#
#
#
#
#
#
#
#    xrdb -merge $sysresources
#
#fi
#
#if [ -f $sysmodmap ]; then
#    xmodmap $sysmodmap
#fi
#
#if [ -f "$userresources" ]; then
#
#
#
#
#
#
#
#    xrdb -merge "$userresources"
#
#fi
#
#if [ -f "$usermodmap" ]; then
#    xmodmap "$usermodmap"
#fi
#
## start some nice programs
#
##twm &
xbindkeys
setxkbmap -layout us,fi
setxkbmap -option 'grp:alt_shift_toggle'
exec i3
source ~/.screenlayout/dual_27_24.sh
