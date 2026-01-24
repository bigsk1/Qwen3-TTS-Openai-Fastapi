#!/bin/bash
# Qwen3-TTS OpenAI-Compatible API Server Startup Script

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8880}
export WORKERS=${WORKERS:-1}
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Check for Python virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Print startup banner
echo ""
echo "░░░░░░░░░░░░░░░░░░░░░░░░"
echo ""
echo "    ╔═╗┬ ┬┌─┐┌┐┌╔═╗  ╔╦╗╔╦╗╔═╗"
echo "    ║═╬╡│││├┤ │││╚═╗───║  ║ ╚═╗"
echo "    ╚═╝└┴┘└─┘┘└┘╚═╝   ╩  ╩ ╚═╝"
echo "    "
echo "    OpenAI-Compatible TTS API"
echo ""
echo "░░░░░░░░░░░░░░░░░░░░░░░░"
echo ""
echo "Starting server on http://$HOST:$PORT"
echo "API Documentation: http://$HOST:$PORT/docs"
echo "Web Interface: http://$HOST:$PORT/"
echo ""

# Start the server
uvicorn api.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
