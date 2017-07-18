FROM centos:7

RUN yum install -y epel-release
RUN yum update -y
RUN yum install -y lighttpd npm
RUN npm install --global gitbook-cli

ENV GITBOOK_VERSION="3.2.2"
RUN gitbook fetch ${GITBOOK_VERSION}

COPY . /var/www/gitbook
WORKDIR /var/www/gitbook
COPY _layouts/plugins/gitbook-plugin-toggle-chapters node_modules

RUN ls
RUN gitbook install && gitbook build

RUN npm cache clear
COPY lighttpd.conf /var/www/gitbook.lighttpd.conf
ENTRYPOINT ["/usr/sbin/lighttpd", "-D", "-f", "/var/www/gitbook.lighttpd.conf"]
