FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

USER root

# The upstream image may contain a stale Yarn apt source without its signing key.
# The Docker-in-Docker devcontainer feature runs apt-get update and fails if that
# source is present. Yarn is not needed for this Python MLOps assignment, so remove
# only source files that reference the Yarn repository before features are installed.
RUN set -eux; \
    for source in /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
        [ -f "$source" ] || continue; \
        if grep -q "dl.yarnpkg.com/debian" "$source"; then \
            rm -f "$source"; \
        fi; \
    done; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i '\|dl.yarnpkg.com/debian|d' /etc/apt/sources.list; \
    fi

USER vscode
