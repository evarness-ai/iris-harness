FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    jq \
    pandoc \
    ca-certificates \
    fonts-dejavu \
    libcairo2 \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    reportlab==4.2.5 \
    weasyprint==62.3 \
    markdown==3.7 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    matplotlib==3.9.2 \
    pypdf==5.1.0 \
    pillow==11.0.0 \
    openpyxl==3.1.5 \
    beautifulsoup4==4.12.3 \
    requests==2.32.3 \
    httpx==0.28.1 \
    lxml==5.3.0 \
    pytz==2024.2 \
    python-dateutil==2.9.0 \
    feedparser==6.0.11 \
    gnews==0.3.7

WORKDIR /workspace

CMD ["/bin/bash"]
