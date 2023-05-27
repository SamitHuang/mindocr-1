import argparse
import yaml

def create_parser():
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', type=str, default='', required=True,
                        help='YAML config file specifying default arguments (default='')')
    parser.add_argument('-o', '--opt', nargs='+',
                        help='Options to change yaml configuration values, e.g. `-o system.distribute=False eval.dataset.dataset_root=/my_path/to/ocr_data`')
    # modelarts
    group = parser.add_argument_group('modelarts')
    group.add_argument('--enable_modelarts', type=bool, default=False,
                       help='Run on modelarts platform (default=False)')
    group.add_argument('--device_target', type=str, default='Ascend',
                       help='Target device, only used on modelarts platform (default=Ascend)')
    # The url are provided by modelart, usually they are S3 paths
    group.add_argument('--multi_data_url', type=str, default='',
                       help='path to multi dataset')
    group.add_argument('--data_url', type=str, default='',
                       help='path to dataset')
    group.add_argument('--ckpt_url', type=str, default='',
                       help='pre_train_model path in obs')
    group.add_argument('--train_url', type=str, default='',
                       help='model folder to save/load')

    #args = parser.parse_args()

    return parser

def _parse_options(opts: list):
    '''
    Args:
        opt: list of str, each str in form f"{key}={value}"
    '''
    options = {}
    if not opts:
        return options
    for opt_str in opts:
        k, v = opt_str.strip().split('=')
        options[k] = yaml.load(v, Loader=yaml.Loader)
    #print('Parsed options: ', options)

    return options

def _merge_options(config, options):
    '''
    Merge options (from CLI) to yaml config.
    '''
    for opt in options: 
        value = options[opt]

        # parse hierarchical key in option, e.g. eval.dataset.dataset_root
        hier_keys = opt.split('.')
        assert hier_keys[0] in config, f'Invalid option {opt}. The key {hier_keys[0]} is not in config.'
        cur = config[hier_keys[0]]
        for level, key in enumerate(hier_keys[1:]):
            if level == len(hier_keys) - 2:
                assert key in cur, f'Invalid option {opt}. The key {key} is not in config.'
                cur[key] = value 
            else:
                assert key in cur, f'Invalid option {opt}. The key {key} is not in config.'
                cur = cur[key] # go to next level
            
    return config

def parse_args_and_config(args=None):
    '''
    Return:
        args: command line argments
        cfg: train/eval config dict 
    '''
    parser = create_parser()
    args = parser.parse_args() # CLI args

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        # TODO: check validity of config arguments to avoid invalid config caused by typo. 
        #_check_cfgs_in_parser(cfg, parser)
        #parser.set_defaults(**cfg)
        #parser.set_defaults(config=args_config.config)
    
    if args.opt:
        options = _parse_options(args.opt)

    cfg = _merge_options(cfg, options)
	
	# TODO: check whether modelarts requires to install addict
    #cfg = Dict(cfg)

    return args, cfg


if __name__ == '__main__':
    args = parse_args()
    print(args.opt)
    options = _parse_options(args.opt)
    print(options)

    # merge options to args

    print(vars(args).keys())
    for k, v in args._get_kwargs():
        print(k, v)


