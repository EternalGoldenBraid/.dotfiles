1. Place in `~/.config/systemd/user/<filename>.service`
2. Run `systemctl --user enable <filename>.service`
3. Run `systemctl --user start <filename>.service`

Troubleshooting:
- `systemctl --user status <filename>.service`
- `journalctl --user -u <filename>.service`
