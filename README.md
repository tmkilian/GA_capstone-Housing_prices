# GA_capstone-Housing_prices
My machine learning capstone projects that predicts rental rates and property values in San Francisco using two datasets (Airbnb and SF Assessor data). I incorporate publicly available geospatial data to improve the accuracy of the models.

There are three notebooks within this repository representing different stages of the project. First, the capstone_data_prev_aws.ipynb notebook was used to load some preliminary datasets and do some data exploration. Some of the datasets are edited slightly in this notebook and there are some preliminary maps of the dataset.

The Baseline_Models_aws.ipynb notebook has a collection of the different baseline models applied to the Airbnb and Assessor's property datasets. These are before geospatial features were added to the datasets.

The crime_311_parks_noise_models_aws.ipynb notebook contains model results after the geospatial data are added and some evalutation plots of the model results.

Much of the data used in the final notebook are included as pickle files in this repository. It should be noted that some of these datasets are large and therefor it may take quite some time to run some of the models or parameter grid searches.

Also included is a pdf of a brief presentation summarizing the study and its results.
