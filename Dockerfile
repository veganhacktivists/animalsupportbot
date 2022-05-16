# syntax=docker/dockerfile:1

FROM python:3.8.10

# Clone repo and install deps
RUN git clone https://github.com/veganhacktivists/animalsupportbot.git
WORKDIR animalsupportbot
RUN pip3 install -r requirements.txt

# Build preload dicts
RUN python3 argmatcher.py

# Grab database and config from cloud storage
# TODO

# Run redditbot
# CMD [ "python3", "redditbot.py", "--run-once"]

# Upload database to cloud storage
# TODO
