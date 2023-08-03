# Define base image/operating system
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install software
#RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda3 -b
RUN rm -f Miniconda3-latest-Linux-x86_64.sh

# Copy files and directory structure to working directory
COPY . . 
#COPY bashrc ~/.bashrc
ENV PATH=/miniconda3/bin:${PATH}

SHELL ["/bin/bash", "--login", "-c"]
RUN conda init bash
#RUN echo 'export PATH=/miniconda3/bin:$PATH' > ~/.bashrc

RUN conda create -n faiss python=3.10
RUN conda install -n faiss -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl 
RUN conda install -n faiss matplotlib scikit-learn
RUN conda run -n faiss pip install h5py


# Run commands specified in "run.sh" to get started

#ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
ENTRYPOINT [ "/bin/bash", "/sisap23-run.sh"]
