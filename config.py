from yacs.config import CfgNode as CN


def get_config(args):
    return CN._load_cfg_py_source(args.config_path)