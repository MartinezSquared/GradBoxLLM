# Use an official Python runtime as a parent image.
FROM python:3.11-slim

# Install uv python manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
COPY ./app_lite /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --frozen
RUN uv run streamlit run streamlit_app_lite.py

EXPOSE 80

# Set environment variables to run Streamlit in headless mode on all interfaces.
ENV STREAMLIT_SERVER_HEADLESS true
ENV STREAMLIT_SERVER_ADDRESS 0.0.0.0
ENV STREAMLIT_SERVER_PORT 80
