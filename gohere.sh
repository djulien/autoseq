#!/bin/bash
echo "to run automatically in Caja file manager set: Edit → Preferences: Behavior"

caja_path() {
    local bookmark_file="$HOME/.config/gtk-3.0/bookmarks"
    [ ! -f "$bookmark_file" ] && bookmark_file="$HOME/.gtk-bookmarks"

    if [ ! -f "$bookmark_file" ]; then
        echo "Error: Bookmark file not found." >&2
        return 1
    fi

    # If no argument is given, list all bookmark names
    if [ -z "$1" ]; then
        echo "Available bookmarks:"
        awk '{print ($2 ? $2 : gensub(".*/", "", "g", $1))}' "$bookmark_file"
        return 0
    fi

    # Find the matching bookmark line (case-insensitive)
    local match
    match=$(grep -i "$1" "$bookmark_file" | head -n 1)

    if [ -z "$match" ]; then
        echo "Error: Bookmark '$1' not found." >&2
        return 1
    fi

    # Extract the URI and remove "file://" scheme
    local uri path
    uri=$(echo "$match" | awk '{print $1}')
    path=$(echo "$uri" | sed -e 's|^file://||' -e 's|%20| |g')

    echo "$path"
}

TOOLS="$(caja_path 'LTC26-dev')" 
export PATH="$TOOLS/Tools26/LightBench/autoseq-git:$PATH"
cd "$(dirname "$0")"
exec bash

goto eof
[Desktop Entry]
Version=1.0
Type=Application
Name=Open Terminal Here
Comment=Open terminal here
#Exec=bash -c 'cd "$(dirname "$(readlink -f "$0" || echo "$A")")"; exec bash'
#Exec=bash -c 'cd "$(dirname "$(readlink -f "$0" || echo "$A")")" && gnome-terminal'
Exec=mate-terminal --working-directory="%f"
Icon=utilities-terminal
Terminal=false
eof: