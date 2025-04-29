FROM php:8.2-cli

RUN apt-get update && apt-get install -y \
    git \
    unzip \
    wget \
    gcc \
    make \
    re2c \
    libpcre3-dev \
    libxml2-dev \
    pkg-config \
    && curl -fsSL https://github.com/derickr/vld/archive/master.zip -o /tmp/vld.zip \
    && unzip /tmp/vld.zip -d /tmp/ \
    && cd /tmp/vld-master \
    && phpize \
    && ./configure \
    && make && make install \
    && echo "extension=vld.so" > /usr/local/etc/php/php.ini \
    && rm -rf /tmp/vld.zip /tmp/vld-master
