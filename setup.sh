#!/bin/bash
# Setup script for coding-agent-eval
# Usage: ./setup.sh [OPTIONS]
#
# Options:
#   --all         Install all dependencies including all LLM providers
#   --minimal     Install only core dependencies (no LLM SDKs)
#   --anthropic   Install with Anthropic SDK only
#   --openai      Install with OpenAI SDK only
#   --dev         Install development dependencies
#   --venv        Create virtual environment first

set -e

echo "======================================"
echo "Coding Agent Evaluation Framework"
echo "======================================"
echo ""

# Parse arguments
INSTALL_TYPE="default"
CREATE_VENV=false

for arg in "$@"; do
    case $arg in
        --all)
            INSTALL_TYPE="all"
            ;;
        --minimal)
            INSTALL_TYPE="minimal"
            ;;
        --anthropic)
            INSTALL_TYPE="anthropic"
            ;;
        --openai)
            INSTALL_TYPE="openai"
            ;;
        --dev)
            INSTALL_TYPE="dev"
            ;;
        --venv)
            CREATE_VENV=true
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all         Install all dependencies"
            echo "  --minimal     Core dependencies only (no LLM SDKs)"
            echo "  --anthropic   Install with Anthropic SDK"
            echo "  --openai      Install with OpenAI SDK"
            echo "  --dev         Development dependencies"
            echo "  --venv        Create virtual environment first"
            echo ""
            echo "Examples:"
            echo "  ./setup.sh --venv --all"
            echo "  ./setup.sh --anthropic"
            exit 0
            ;;
    esac
done

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 10 ]); then
    echo "Error: Python 3.10+ required (found $python_version)"
    exit 1
fi
echo "  ✓ Python $python_version"
echo ""

# Create virtual environment if requested
if [ "$CREATE_VENV" = true ]; then
    echo "Creating virtual environment..."
    if [ -d "venv" ]; then
        echo "  Virtual environment already exists"
    else
        python3 -m venv venv
        echo "  ✓ Created venv/"
    fi
    
    echo ""
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "  ✓ Activated"
    echo ""
fi

# Install dependencies based on type
echo "Installing dependencies ($INSTALL_TYPE)..."
echo ""

case $INSTALL_TYPE in
    all)
        pip install -e ".[all]"
        ;;
    minimal)
        pip install -e .
        ;;
    anthropic)
        pip install -e ".[anthropic]"
        ;;
    openai)
        pip install -e ".[openai]"
        ;;
    dev)
        pip install -e ".[dev,all]"
        ;;
    default)
        # Default: install from requirements.txt
        pip install -r requirements.txt
        pip install -e .
        ;;
esac

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""

# Check API keys
echo "API Key Status:"
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "  ✓ ANTHROPIC_API_KEY is set"
else
    echo "  ✗ ANTHROPIC_API_KEY not set"
    echo "    Run: export ANTHROPIC_API_KEY=your-key-here"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo "  ✓ OPENAI_API_KEY is set"
else
    echo "  ✗ OPENAI_API_KEY not set (optional)"
fi

if [ -n "$GROQ_API_KEY" ]; then
    echo "  ✓ GROQ_API_KEY is set"
else
    echo "  ✗ GROQ_API_KEY not set (optional)"
fi

echo ""
echo "Quick Start:"
echo "  # Run a simple test"
echo "  python test_e2e.py --skip-agent"
echo ""
echo "  # Run benchmark"
echo "  python benchmark.py --tasks eval/tasks/ --models anthropic:claude-sonnet-4-20250514"
echo ""
echo "  # Or use make"
echo "  make check-env"
echo "  make test-imports"
echo ""

if [ "$CREATE_VENV" = true ]; then
    echo "Note: Virtual environment is active. To deactivate, run: deactivate"
    echo "      To reactivate later, run: source venv/bin/activate"
fi
