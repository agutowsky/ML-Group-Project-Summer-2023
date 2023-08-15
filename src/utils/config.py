import yaml

def load_config(src: str):
    '''
    Load configuration data from a YAML file.

    Args:
        src (str): The file path to the YAML configuration file.
    
    Returns:
        dict or None: The loaded config data as a dict, or None if the file could not be read.
    '''
    config: None
    with open(src, "r") as yaml_file :  config = yaml.safe_load(yaml_file)
    return config