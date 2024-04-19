#!/bin/bash


_install_resim_cli() {
    curl -L https://github.com/resim-ai/api-client/releases/latest/download/resim-linux-amd64 -o /usr/bin/resim
    chmod +x /usr/bin/resim
}

_su_install_resim_cli() {
    # If we have sudo, use it
    if command -v sudo 
    then
	s="sudo"
    fi
    eval "${s} bash -c \"$(declare -f _install_resim_cli); _install_resim_cli\""
}

which resim 1>/dev/null || _su_install_resim_cli
