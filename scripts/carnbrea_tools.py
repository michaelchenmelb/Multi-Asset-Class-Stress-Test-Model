# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import subprocess
import logging
import os

def update_package(list_packages):
    
    logger = logging.getLogger(__name__)
    logger.info('checking package')
    reqs = subprocess.check_output([sys.executable, '-m','pip','freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    
    list_uninstalled_packages = [_package for _package in list_packages if _package not in installed_packages]
    
    if len(list_uninstalled_packages) > 0:
        
        logger.info('missing required package {}'.format('\n'.join(list_uninstalled_packages)))
        
        try:
            subprocess.call([r"\\cbsvr01\company data\_Carnbrea Asset Management\9. Stress Testing\Model\Scripts (DO NOT MOVE THIS FOLDER)\install_package.bat"])
            logger.info('calling package installation batch successully')
        
        except Exception as e:
            logger.error('cant install packages required. error: {}'.format(e))
        
    else:
        
        logger.info('required packages satisfied')
        
    return

def start_logger(path_log_file):
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    dir_path = os.path.dirname(path_log_file)
    
    if not os.path.exists(dir_path):
        print ("creating log file: %s"%path_log_file)
        os.makedirs(dir_path)
    
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers = [logging.FileHandler(path_log_file),
                                    logging.StreamHandler(sys.stdout)])
    
    return
        
if __name__ == "__main__":
    
    pass
    
    
