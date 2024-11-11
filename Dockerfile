FROM cityflowproject/cityflow:latest

COPY ./requirements.txt /requirements.txt

#Dependencies for open-CV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install \
    jupyter \
    jupyterlab \
    ipykernel

WORKDIR /repo/

EXPOSE 8888

CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--port=8888"]
