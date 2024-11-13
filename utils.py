# config_loader.py
import yaml


# Define the range constructor
def range_constructor(loader, node):
    """Constructor for !range tag in YAML"""
    value = loader.construct_sequence(node)
    start, end, step = value
    return list(range(start, end, step))


# Create a new YAML loader class with our constructor
class RangeLoader(yaml.SafeLoader):
    pass


# Add the constructor to our loader class
RangeLoader.add_constructor("!range", range_constructor)


# Use this function to load the config
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=RangeLoader)
