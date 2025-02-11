#!/bin/bash

# Script to find uv path, add its directory to PATH, and install uv if not found
# Includes macOS/brew installation option

os_type=$(uname -s)

# --- Check if uv is already installed and working ---
if uv --version >/dev/null 2>&1; then
  echo "uv is already installed and working (verified by 'uv --version')."
  # Find uv path even if it's working, to ensure PATH is set correctly
  uv_path=$(which uv)
  if [ -z "$uv_path" ]; then
    uv_path=$(command -v uv)
  fi
else
  echo "uv is not working or not found (uv --version failed)."

  # --- Attempt to install uv based on OS ---
  if [[ "$os_type" == "Darwin" ]]; then # macOS detection
    echo "Operating system detected: macOS"
    echo "Attempting to install uv using: brew install uv"
    if brew install uv; then
      echo "uv installation attempted successfully using brew."
      # Re-check if uv is working after brew installation attempt
      if uv --version >/dev/null 2>&1; then
        echo "uv is now working after brew installation (verified by 'uv --version')."
        uv_path=$(which uv)
        if [ -z "$uv_path" ]; then
          uv_path=$(command -v uv)
        fi
      else
        echo "uv installation via brew might have failed or uv is still not working after installation attempt (uv --version still fails)."
        echo "Please check the brew installation output for errors and ensure uv is installed correctly."
        fallback_install=true # Fallback to curl install if brew fails on macOS
      fi
    else
      echo "uv installation using brew failed. Please check the brew output for errors."
      fallback_install=true # Fallback to curl install if brew fails on macOS
    fi

    if [[ "$fallback_install" == "true" ]]; then
      echo "Falling back to curl installation method..."
      echo "Attempting to install uv using: curl -LsSf https://astral.sh/uv/install.sh | sh"
      install_command="curl -LsSf https://astral.sh/uv/install.sh | sh"
      if bash -c "$install_command"; then
        echo "uv installation attempted successfully using curl script (fallback)."
        if uv --version >/dev/null 2>&1; then
          echo "uv is now working after curl script installation (verified by 'uv --version')."
          uv_path=$(which uv)
          if [ -z "$uv_path" ]; then
            uv_path=$(command -v uv)
          fi
        else
          echo "uv installation via curl script might have failed or uv is still not working after installation attempt (uv --version still fails)."
          echo "Please check the installation output for errors and ensure uv is installed correctly."
          exit 1
        fi
      else
        echo "uv installation using curl script (fallback) also failed. Please check the output for errors."
        exit 1
      fi
    fi


  else # Not macOS - use curl install script
    echo "Operating system detected: $(uname -s). Using curl installation method."
    echo "Attempting to install uv using: curl -LsSf https://astral.sh/uv/install.sh | sh"
    install_command="curl -LsSf https://astral.sh/uv/install.sh | sh"
    if bash -c "$install_command"; then
      echo "uv installation attempted successfully using curl script."
      if uv --version >/dev/null 2>&1; then
        echo "uv is now working after curl script installation (verified by 'uv --version')."
        uv_path=$(which uv)
        if [ -z "$uv_path" ]; then
          uv_path=$(command -v uv)
        fi
      else
        echo "uv installation via curl script might have failed or uv is still not working after installation attempt (uv --version still fails)."
        echo "Please check the installation output for errors and ensure uv is installed correctly."
        exit 1
      fi
    else
      echo "uv installation using curl script failed. Please check the output for errors."
      exit 1
    fi
  fi
fi

# --- Find uv path (if not already found or if installation was attempted) ---
if [ -z "$uv_path" ]; then
  uv_path=$(which uv)
  if [ -z "$uv_path" ]; then
    uv_path=$(command -v uv)
    if [ -z "$uv_path" ]; then
      echo "Error: Could not determine uv executable path even after installation attempt."
      echo "Please ensure uv is correctly installed and in your system's PATH."
      exit 1
    fi
  fi
fi


# Extract the directory containing the uv executable
uv_dir=$(dirname "$uv_path")

# Check if the directory exists
if [ ! -d "$uv_dir" ]; then
  echo "Error: Directory containing uv ('$uv_dir') does not exist."
  echo "The path found for uv might be invalid."
  exit 1
fi

# Add the uv directory to the beginning of the PATH environment variable
export PATH="$uv_dir:$PATH"

# Optional: Print the updated PATH for verification
echo "uv directory '$uv_dir' added to PATH."
echo "Updated PATH: $PATH"

# Optional: Verify if uv is now directly accessible (for testing)
if uv --version >/dev/null 2>&1; then
  echo "uv command is now accessible directly (verified by 'uv --version')."
else
  echo "uv command might still not be directly accessible in all contexts."
  echo "   If you still have issues, ensure your shell environment is properly configured."
fi

exit 0
