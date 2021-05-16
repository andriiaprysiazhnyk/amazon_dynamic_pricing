import os
import yaml


def copy_config(config, log_path):
    with open(os.path.join(log_path, "agent_config.yaml"), "w") as fp:
        yaml.dump(config, fp)


def create_sub_folder(folder, sub_folder):
    path = os.path.join(folder, sub_folder)
    os.mkdir(path)
    return path
