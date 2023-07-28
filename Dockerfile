# Use an official Miniconda image as a parent image
FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV APP_HOME /app

# Set the working directory to /app
WORKDIR $APP_HOME

# Add the current directory contents to the container at /app
COPY src/ $APP_HOME/src/
COPY environment.yml $APP_HOME


# Creates a conda environment from the environment.yml file
RUN conda env create -f environment.yml && conda clean --all --yes || exit 1

# Make run commands use the new environment
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH

# Make the container executable
ENTRYPOINT ["/bin/bash", "-c"]

CMD ["source activate ChemLactica && python train.py"]
