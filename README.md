# 3D Troposphere


## WARNING: CODE IN DEVELOPMENT


#### Status

- Dataset - based on file location is config, a dataset is made from the ERA5

- Model - base model built, requires user to define a model class and provide its path within config

- Point Cloud - generator made and saves uniformly sampled points to ply file

- Render - this code has not been developed 



I tried to make the code as clear as possible for you to understand. I tried to be as mnodular and plug and play as i could.

requirements.txt made using pip-chill 
```
pip install pip-chill
pip-chill > requirements.txt

python main.py --config configs/config.yaml  
```

# 3D Troposphere

Transform discrete observations of atmospheric data into a continuous volumetric representation


## Training

To train your own model on ERA5 data for your own use cases
1. Clone the repo `git clone https://github.com/mickyals/3D_Troposphere.git`
2. Open the directory `cd 3D_Troposphere`
3. Create the environment `conda create --name 3D_Troposphere python=3.11` and activate `conda activate 3D_Troposphere`
4. Install the required packages `pip install -r requirements.txt`
5. Process the ERA5 nc file using `insert code here from EDA ipynb`
6. Define a pytorch IterableDataset, an example can be found in the `data` directory.
7. Create a configuration file following the format defined with `configs/config.yaml`
8. Train your model `python main.py --config configs/config.yaml`


## Rendering
After your model has trained and convered, you will want to create the volumetric representation of your data. 
1. Define the path to your saved weights within the config yaml at `model_checkpoint`
2. Define the sampling ranges within `pointcloud` within the yaml
3. Sample from your trained model `code for sampling goes here`
4. Render the saved point cloud file using the tools `TBD`