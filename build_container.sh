#! /bin/bash

docker run -it -v /var/run/docker.sock:/var/run/docker.sock -v `pwd`:/config merc1031/docker-composer:01.20.19 -f /config/docker-compose.yml build

docker tag merc1031/cambilight:latest merc1031/cambilight:01.20.19
