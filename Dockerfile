# Use an official Miniconda image as a parent image
FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV APP_HOME /app

# Set the working directory to /app
WORKDIR $APP_HOME

# Add the current directory contents to the container at /app
COPY src/ $APP_HOME/src/
COPY requirements.txt $APP_HOME
COPY .small_data $APP_HOME/.small_data
# install anaconda
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean && rm -rf /var/lib/apt/lists/*

RUN /usr/bin/python -m pip install --upgrade pip
RUN pip install -r $APP_HOME/requirements.txt

# Make run commands use the new environment
# CMD ["python3","-m","unittest", "src/tests/precommit_test.py"]
CMD ["export", 'CUDA_VISIBLE_DEVICES="0, 1"']
CMD ["nvidia-smi"]
CMD ["export", "TOKENIZERS_PARALLELISM=false"]
CMD ["python3", "src/train.py", "--from_pretrained", "facebook/galactica-125m", "--model_config", "125m",  "--training_data_dir", ".small_data/train", "--valid_data_dir", ".small_data/valid", "--max_steps", "10", "--eval_steps", "10", "--save_steps", "10", "--experiment_name", "gal125m_dockertest", "--checkpoints_root_dir", "./checkpoints", "--track_dir", "./aim"]
