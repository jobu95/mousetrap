#!/bin/bash

set -o errexit
set -o xtrace

protoc -I . --python_out . frames.proto

