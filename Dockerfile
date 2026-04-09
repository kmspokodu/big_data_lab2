FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_LEVEL=INFO \
    RUN_MODE=api

WORKDIR /app

# Копируем requirements 
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Копируем весь проект
COPY . .

# Создаем необходимые директории
RUN mkdir -p /app/data /app/experiments /app/logs

# Устанавливаем права доступа
RUN chmod +x src/*.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os; assert os.path.isfile('experiments/log_reg.sav'), 'Model not found'" || exit 1

# Use an entrypoint script to decide what to run
ENTRYPOINT ["bash", "-c", "if [ \"$RUN_MODE\" = \"pipeline\" ]; then python src/preprocess.py && python src/train.py && python src/predict.py -m LOG_REG -t smoke; else python src/api.py; fi"]