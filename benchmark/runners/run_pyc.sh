#!/bin/bash

if [ "$1" == "compile" ]; then
    if [[ "$2" == *.py ]]; then
        pyc build "$2" -o "${2%.py}.out"
    elif [[ "$2" == *.cu ]]; then
        pyc kernel register "$2"
        pyc build "$2" -o "${2%.cu}.out"
    fi
elif [ "$1" == "run" ]; then
    if [[ "$2" == *.py ]]; then
        # shellcheck disable=SC2046
        ./$(basename "${2%.py}.out")
    elif [[ "$2" == *.cu ]]; then
        # shellcheck disable=SC2046
        ./$(basename "${2%.cu}.out")
    fi
else
    echo "Usage: $0 {compile|run} <workload>"
    exit 1
fi
