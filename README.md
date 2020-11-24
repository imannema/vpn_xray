For this project we dive into the world of network traffic information and ultimately build a classifier that determines if network traffic data involves video streaming or not.

To begin, build the docker image which establishes the environment that contains all necessary imports for this project. To do so, either build the docker image with the dockerfile provided, or see the built image on my docker hub: https://hub.docker.com/repository/docker/imannema/projimg

Once your environment is set, get your data file you want to test and place it in the same directory as run.py and data. The data folder contains all training data for the model and run.py is where the training and prediction is performed.

To test the model on your dataset, type 'python run.py <file.csv>', where file.csv is the input data you want to classify; make sure to NOT surround your csv file with quotes.

The output has log messages explaining which step the process is in. Cleaning and training the data may take some time. Please ignore warning messages. The output will display if the model predicts your input is streaming or not; True for streaming, False if not.

The project currently is trained and tested on 2 minute chunks of data. Preferably input 2 minute chunks to test; I am currently working on making the project work for larger input sizes.
