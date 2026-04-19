FROM python:3.10

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=20

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
	pip install --no-cache-dir --timeout 1000 --retries 20 -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
