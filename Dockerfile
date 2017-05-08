FROM nginx:latest

ENV GITBOOK_VERSION="3.2.2"

RUN apt-get update \
    && apt-get install -y curl git bzip2 libfontconfig1-dev xz-utils gnupg
RUN curl -sL https://deb.nodesource.com/setup_4.x | bash -
RUN apt-get install -y nodejs

RUN npm install --global gitbook-cli \
    && gitbook fetch ${GITBOOK_VERSION} \
    && npm cache clear \
    && rm -rf /tmp/*

COPY . /gitbook
WORKDIR /gitbook

RUN gitbook install && gitbook build 

RUN rm -rf /usr/share/nginx/html/*
RUN cp -r ./_book/* /usr/share/nginx/html/

