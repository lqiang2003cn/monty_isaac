import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/lq/lqtech/dockers/monty_isaac/install/monty_demo'
