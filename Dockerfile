FROM ghcr.io/lcogt/banzai:1.19.1-6-g8fe8f7e2

USER root

COPY --chown=10087:10000 . /lco/banzai-floyds

RUN apt-get -y update && apt-get -y install gcc && \
        pip install --no-cache-dir /lco/banzai-floyds/ && \
        apt-get -y remove gcc && \
        apt-get autoclean && \
        rm -rf /var/lib/apt/lists/*

USER archive
