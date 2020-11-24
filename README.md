For this project we dive into the world of network traffic information and ultimately build a classifier that determines if data involves video streaming or not.

To begin, build the docker image which establishes the environment which all necessary imports for this project. To do so, wither build the docker image with the dockerfile provided, or see the image on my docker hub: https://hub.docker.com/repository/docker/imannema/projimg

Once your environment is set, get your data file you want to test and place it in the same directory as run.py and data. The data folder contains all training data for the model and run.py is where the training and prediction is performed.

To test the model on your dataset, type 'python run.py <filename>.csv', where filename is the name our your csv file, make sure to not surround your csv file with quotes.

The output has log messages explaining which step the process is in. Please ignore warning messages.

The project currently is trained and tested on 2 minute chunks of data. Preferably input 2 minute chunks to test; I am currently working on making the project work for larger input sizes.
