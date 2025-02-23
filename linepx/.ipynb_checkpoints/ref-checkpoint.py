import os
from pathlib import Path
#原来的代码
# root_dir = Path(os.getcwd()) / '..'
# data_root = root_dir / 'data'
#修改后的代码
ssh_dir=Path(os.getcwd()) / '..'/'..'/'..'
data_root = ssh_dir / 'data'

ext = '.pkl'
input_size = 320
data_folder = data_root / 'linepx' / 'processed'

cacheFile = data_root / 'train_cache.ptar'
