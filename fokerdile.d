FROM python:3.8-slim-buster
COPY doferkile.py /
RUN pip install flask
# ENV
WORKDIR "$PWD"
# VOLUME
CMD [ "python", "./doferkile.py" ]