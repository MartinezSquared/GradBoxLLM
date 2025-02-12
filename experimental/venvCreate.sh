#!/bin/bash

# Script to create a uv virtual environment and install project dependencies from pyproject.toml

# --- Check if pyproject.toml exists ---
if [ ! -f "pyproject.toml" ]; then
  echo "Error: pyproject.toml file not found in the current directory."
  echo "Please run this script from the root directory of your Python project."
  exit 1
fi

# --- Create virtual environment using uv ---
echo "Creating a uv virtual environment..."
if uv venv; then
  echo "✅ uv virtual environment created successfully in .venv directory."
else
  echo "❌ Failed to create uv virtual environment."
  echo "Please check for errors above and ensure uv is installed and working."
  exit 1
fi

# --- Activate virtual environment instructions ---
echo "\n--- Virtual Environment Activation ---"
echo "Virtual environment created in '.venv' directory."
echo "To activate the virtual environment, use the appropriate command for your shell:"
echo ""
echo "  For macOS/Linux (bash, zsh):  source .venv/bin/activate"
echo "  For Windows (Command Prompt):    .venv\\Scripts\\activate.bat"
echo "  For Windows (PowerShell):       .venv\\Scripts\\Activate.ps1"
echo ""
echo "Make sure to activate the virtual environment before installing dependencies and running your project."

# --- Install dependencies from pyproject.toml using uv ---
echo "\n--- Installing project dependencies from pyproject.toml using uv... ---"
if uv pip install .; then
  echo "✅ Project dependencies installed successfully using 'uv pip install .'!"
  echo ""
  echo "🎉  Project setup complete!"
  echo "   - Virtual environment created in '.venv'."
  echo "   - Dependencies installed from 'pyproject.toml'."
  echo ""
  echo "   Remember to activate the virtual environment before working on your project."
else
  echo "❌ Failed to install project dependencies using 'uv pip install .'."
  echo "Please check for errors above and ensure uv is working correctly and your pyproject.toml is valid."
  exit 1
fi

exit 0
