from accelerate import Accelerator

accelerator = Accelerator(mixed_precision='no')
accelerator.print(f"Total devices used: {accelerator.num_processes}")
accelerator.print(f'Current device: {str(accelerator.device)}')
accelerator.print(f'Current process index: {accelerator.process_index}')

# import os
# from accelerate.utils import write_basic_config
# write_basic_config() # Write a config file
# os._exit(0) # Restart the notebook to reload info from the latest config file 
