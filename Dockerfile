# syntax=docker/dockerfile:1

FROM python:3.8.10

# Clone repo and install deps
RUN git clone https://github.com/veganhacktivists/animalsupportbot.git
WORKDIR animalsupportbot
RUN pip3 install -r requirements.txt

# Build preload dicts
RUN python3 argmatcher.py

# Run redditbot
# Expects log_db.json and config.yaml to be in /asb_files/
CMD [ "python3", "redditbot.py", "--run-once", "--log-db",  "/asb_files/log_db.json", "--config", "/asb_files/config.yaml" ] 
