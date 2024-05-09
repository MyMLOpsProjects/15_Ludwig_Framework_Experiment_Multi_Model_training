# import logging
# import pandas as pd
# from ludwig.api import LudwigModel
# from ludwig.visualize import confusion_matrix
# from ludwig.hyperopt.run import hyperopt

# df = pd.read_csv("./titanic.csv")
# lm = LudwigModel("config.yml",logging_level=logging.INFO)
# a,b,c = lm.train(dataset=df,output_directory="Experiment_logs",skip_save_processed_input=True)
#!/usr/bin/env python

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/latest/examples/titanic/).

# Import required libraries
import logging
import os
import shutil

import yaml

from ludwig.api import LudwigModel
from ludwig.datasets import titanic
from ludwig.visualize import learning_curves

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

# Download and prepare the dataset
training_set, test_set, _ = titanic.load(split=True)

models = ["model_1","model_2"]
list_of_train_stats = []

for model in models:
  # Define Ludwig model object that drive model training
  lm = LudwigModel(config="{}".format(model)+".yml", logging_level=logging.INFO)
  # initiate model training
  (
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
  ) = lm.train(
    dataset=training_set, experiment_name="simple_experiment", model_name="simple_model", skip_save_processed_input=True
  )
  list_of_train_stats.append(train_stats)
  print(">>>>>>> completed: ", model, "\n")

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
  print("\t", item)

# # batch prediction
# model.predict(test_set, skip_save_predictions=False)

# generating learning curves from training
learning_curves(
    list_of_train_stats,
    "Survived",
    model_names=models,
    output_directory="./visualizations",
    file_format="png",
)