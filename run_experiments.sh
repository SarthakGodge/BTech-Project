#!/usr/bin/env bash
# Kept for compatibility: the single command now lives in run_all.sh.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_all.sh" "$@"
