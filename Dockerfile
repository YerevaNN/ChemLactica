# Use an official Miniconda image as a parent image
FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV APP_HOME /app

# Set the working directory to /app
WORKDIR $APP_HOME

# Add the current directory contents to the container at /app
COPY src/ $APP_HOME/src/
COPY requirements.txt $APP_HOME

# install anaconda
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -r $APP_HOME/requirements.txt

# Make run commands use the new environment
CMD ["python3", "src/hello_world.py"]
