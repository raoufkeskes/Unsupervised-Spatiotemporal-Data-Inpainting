import yaml
import os

root_project = os.path.dirname ( os.path.dirname(os.path.abspath(__file__)) )


def get_config():
    try:
        with open(str(root_project) + "/Config/configuration.yaml", 'r') as file:
            config = yaml.safe_load(file)
            return config

    except Exception as e:
        raise Exception('Error reading the config file : ' + str(e))



# add project root directory to config yaml file if it does not exist
#             if "root" not in config["global"]:
#                 config["global"]["root"] = root_project
#                 yaml.dump(config)