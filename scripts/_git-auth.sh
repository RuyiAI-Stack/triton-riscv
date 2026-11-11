#!/usr/bin/env bash
#
# Sourced helper (not meant to be executed directly).
#
# Anonymous git from some self-hosted runners gets a 401, and the build clones
# things: setup.sh clones triton, and triton's cmake clones googletest. Hand git
# the token the way actions/checkout does, through a throwaway global config.
#
# Usage:
#   source scripts/_git-auth.sh

if [ -n "${GITHUB_TOKEN:-}" ]; then
    _gitauth="$(mktemp)"
    printf '[http "https://github.com/"]\n\textraheader = AUTHORIZATION: basic %s\n' \
        "$(printf 'x-access-token:%s' "${GITHUB_TOKEN}" | base64 | tr -d '\n')" > "${_gitauth}"
    export GIT_CONFIG_GLOBAL="${_gitauth}"
    unset _gitauth
fi
