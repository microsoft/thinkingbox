#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -e

#######################################################################
# ThinkingBox Background Tasks Manager
#
# This script manages the background services required for the ThinkingBox
# AI training environment. It performs the following functions:
#
# 1. Starts and monitors the tb mcp-start if not already running
# 3. Starts and monitors the typesense-server if not already running
# 4. Tails the log files of all services for real-time monitoring
# 5. Handles graceful shutdown of all started processes when terminated
# 6. Provides process status information during execution
#
# The script is designed to run continuously until interrupted with Ctrl+C,
# at which point it will properly clean up all processes it started.
#
# Usage:
#   ./background_tasks.sh
# Environment variables:
#   TYPESENSE_API_KEY: API key for the typesense server, default: "Fake"
#
#######################################################################

if [[ -z "$TYPESENSE_API_KEY" ]]; then
    TYPESENSE_API_KEY="Fake"
fi

if [[ -z "$TB_MCP_START_SERVERS_FILE" ]]; then
    echo "Environment variable TB_MCP_START_SERVERS_FILE not set; running without a servers file."
fi

check_exe() {
    local exe_path=$(which "$1" 2>/dev/null)
    if [[ -z "$exe_path" ]]; then
        echo "Executable '$1' does not exist in PATH."
        exit 1
    else
        echo "Found $1 -> $exe_path"
    fi
}

# check required executables are in PATH
check_exe tb mcp-start
check_exe typesense-server

# Define log file paths
MCP_LOG="/tmp/thinkingbox-mcp-server.log"
TYPESENSE_LOG="/tmp/typesense.log"

# Function to check if a process is running by pattern
is_process_running() {
    local pattern="$1"
    pgrep -f "$pattern" > /dev/null
    return $?
}

# Function to get PID of a process by pattern
get_process_pid() {
    local pattern="$1"
    pgrep -f "$pattern" | head -1
}

# Track processes we start
MCP_WE_STARTED=""
TYPESENSE_WE_STARTED=""
MCP_TAIL_PID=""

# Function to clean up on exit
cleanup() {
    echo "Cleaning up processes..."

    # Stop tail processes
    if [ -n "$MCP_TAIL_PID" ]; then
        kill $MCP_TAIL_PID 2>/dev/null
    fi

    # Stop main processes if we started them
    if [ -n "$MCP_WE_STARTED" ]; then
        local mcp_pid=$(get_process_pid "tb mcp-start")
        if [ -n "$mcp_pid" ]; then
            echo "Stopping tb mcp-start (PID: $mcp_pid)"
            kill $mcp_pid 2>/dev/null
        fi
    fi

    if [ -n "$TYPESENSE_WE_STARTED" ]; then
        local typesense_pid=$(get_process_pid "typesense-server")
        if [ -n "$typesense_pid" ]; then
            echo "Stopping typesense-server (PID: $typesense_pid)"
            kill $typesense_pid 2>/dev/null
        fi
    fi

    exit 0
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Start typesense-server if not already running
if is_process_running "typesense-server"; then
    echo "typesense-server is already running with PID: $(get_process_pid 'typesense-server')"
else
    mkdir -p /tmp/typesense
    echo "$TYPESENSE_API_KEY" > /tmp/typesense/.api_key
    mkdir -p /tmp/typesense/data
    echo "Starting typesense-server..."
    typesense-server --data-dir="/tmp/typesense/data" --api-key="$TYPESENSE_API_KEY" --enable-cors > "$TYPESENSE_LOG" 2>&1 &
    TYPESENSE_WE_STARTED=1
    echo "typesense-server started with PID: $!"
fi

# Wait for typesense /health to return 200 before starting mcp-start.
# Cold start can take minutes while embedder models load; mcp-start sessions
# that hit Typesense before it's ready will 503 their way to session failure.
echo "Waiting for typesense-server to become ready..."
for _ in $(seq 1 150); do
    if curl -sf http://localhost:8108/health >/dev/null 2>&1; then
        echo "typesense-server is ready."
        break
    fi
    sleep 2
done
if ! curl -sf http://localhost:8108/health >/dev/null 2>&1; then
    echo "typesense-server did not become ready within 5 minutes; aborting."
    exit 1
fi

# Start tb mcp-start if not already running
if is_process_running "tb mcp-start"; then
    echo "tb mcp-start is already running with PID: $(get_process_pid 'tb mcp-start')"
else
    echo "Starting tb mcp-start..."
    if [[ -n "$TB_MCP_START_SERVERS_FILE" ]]; then
        if [[ -r "$TB_MCP_START_SERVERS_FILE" ]]; then
            echo "Using servers file: $TB_MCP_START_SERVERS_FILE"
            tb mcp-start --port 7111 --host 0.0.0.0 --servers "$TB_MCP_START_SERVERS_FILE" > "$MCP_LOG" 2>&1 &
        else
            echo "Error: TB_MCP_START_SERVERS_FILE is set to '$TB_MCP_START_SERVERS_FILE',"
            echo "but the file does not exist or is not readable. Aborting tb mcp-start."
            exit 1
        fi
    else
        echo "No servers file provided, starting without it."
        tb mcp-start --port 7111 --host 0.0.0.0 > "$MCP_LOG" 2>&1 &
    fi
    MCP_WE_STARTED=1
    echo "tb mcp-start started with PID: $!"
fi

# Start tailing the MCP log
tail -f "$MCP_LOG" &
MCP_TAIL_PID=$!

echo "All processes are running. Press Ctrl+C to stop."

# Wait indefinitely
wait
