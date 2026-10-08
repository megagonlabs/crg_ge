#!/usr/bin/env bash

set -euo pipefail

passes=3
while getopts ":n:" option; do
    case "$option" in
        n) passes=$OPTARG ;;
        *)
            echo "Usage: $0 [-n passes] <folder-or-yaml-file> [batch_run_ce-args ...]" >&2
            exit 2
            ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [-n passes] <folder-or-yaml-file> [batch_run_ce-args ...]" >&2
    exit 2
fi
if ! [[ $passes =~ ^[1-9][0-9]*$ ]]; then
    echo "Pass count must be a positive integer: $passes" >&2
    exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

target=$1
shift
if [[ ! -d "$target" && ! -f "$target" ]]; then
    echo "Folder or file does not exist: $target" >&2
    exit 1
fi

for ((pass = 1; pass <= passes; pass++)); do
    printf 'Starting pass %d of %d\n' "$pass" "$passes"
    while IFS= read -r -d '' yaml_file; do
        printf 'Running uv batch_run_ce for %s (pass %d of %d)\n' "$yaml_file" "$pass" "$passes"
        if [[ $# -gt 0 ]]; then
            uv run batch_run_ce "$yaml_file" --resume "$@"
        else
            uv run batch_run_ce "$yaml_file" --resume
        fi
    done < <(find "$target" -type f \( -name '*.yaml' -o -name '*.yml' \) -print0)
done
