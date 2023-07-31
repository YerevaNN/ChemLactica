# Use an official Miniconda image as a parent image
FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV APP_HOME /app

# Set the working directory to /app
WORKDIR $APP_HOME

# Add the current directory contents to the container at /app
COPY src/ $APP_HOME/src/
COPY environment.yml $APP_HOME

# Creates a conda environment from the environment.yml file

# install anaconda
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean && rm -rf /var/lib/apt/lists/* && \
        wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
        /bin/bash ~/miniconda.sh -b -p /opt/conda && \
        rm ~/miniconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH
ENV PYTHONPATH /opt/conda/lib/python3.8/site-packages

# setup conda virtual environment
RUN conda update conda && \
    conda update -n base -c defaults conda && \
    conda env create --name ChemLactica -f environment.yml

RUN pip install /usr/local/lib/python3.8/dist-packages/torch
# Make run commands use the new environment
RUN echo "conda activate ChemLactica" > ~/.bashrc
ENV PATH /opt/conda/envs/ChemLactica/bin:$PATH
ENV CONDA_DEFAULT_ENV ChemLactica

CMD ["python3", "src/hello_world.py"]
