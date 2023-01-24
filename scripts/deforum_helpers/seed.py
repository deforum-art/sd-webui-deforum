import random

def next_seed(args):
    if args.seed_behavior == 'iter':
        if args.seed_internal % args.seed_iter_N == 0:
            args.seed += 1
        args.seed_internal += 1
    elif args.seed_behavior == 'ladder':
        if args.seed_internal == 0:
            args.seed += 2
            args.seed_internal = 1
        else:
            args.seed -= 1
            args.seed_internal = 0
    elif args.seed_behavior == 'alternate':
        if args.seed_internal == 0:
            args.seed += 1
            args.seed_internal = 1
        else:
            args.seed -= 1
            args.seed_internal = 0
    elif args.seed_behavior == 'fixed':
        pass # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32 - 1)
    return args.seed