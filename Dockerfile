FROM ghcr.io/lcogt/banzai:1.22.0

USER root

RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /lco/banzai-floyds/

RUN poetry install --directory=/lco/banzai-floyds -E cpu --no-root --no-cache

COPY . /lco/banzai-floyds

RUN poetry install --directory /lco/banzai-floyds -E cpu --no-cache

USER archive
