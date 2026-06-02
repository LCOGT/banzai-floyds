FROM ghcr.io/lcogt/banzai:1.36.2

USER root

ENV UV_PROJECT_ENVIRONMENT=/lco/banzai/.venv

RUN mkdir -p /home/archive/.cache/matplotlib

COPY pyproject.toml uv.lock /lco/banzai-floyds/

RUN uv sync --locked --directory=/lco/banzai-floyds --no-install-project

COPY pytest.ini /home/archive/pytest.ini

RUN uv sync --locked --directory /lco/banzai-floyds

USER archive
