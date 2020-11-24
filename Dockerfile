FROM ucsdets/datascience-notebook:2020.2-stable

#LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

USER root

RUN pip install --no-cache-dir scikit-learn

USER $NB_UID

# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]
