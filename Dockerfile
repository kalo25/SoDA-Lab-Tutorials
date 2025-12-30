FROM python:3.12-slim

# Set working directory
WORKDIR /project

# Copy project files into container
COPY . /project

# Install dependencies
# Option A: install from a pinned requirements.txt you maintain
# COPY requirements.txt /project/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Option B (teaching convenience): install a minimal set directly
RUN pip install --no-cache-dir numpy pandas matplotlib statsmodels

# Run the analysis script
CMD ["python", "reproducibility.py"]
