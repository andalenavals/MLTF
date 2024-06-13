import os
import MLTF
import pickle

import logging
logger = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Recpickling from SHE_KERAS to MLTF')
 
    parser.add_argument('--files',
                        default=[],
                        nargs='+',
                        help='Features for training g')


    args = parser.parse_args()
    return args

       
def update_normer(filename):
    logger.info("changing %s"%(filename))
    assert os.path.isfile(filename)
    with open(filename, 'rb') as handle:
        normer = pickle.load(handle)

    newnormer=MLTF.normer.Normer(a=normer.a, b=normer.b, type=normer.type)
    with open(filename, 'wb') as handle:
        pickle.dump(newnormer, handle, -1)
    

def main():    
    args = parse_args()
    loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
    #logging.basicConfig(format=loggerformat, level=logging.DEBUG)
    logging.basicConfig(format=loggerformat, level=logging.INFO)
    #logging.basicConfig(format=loggerformat, level=logging.NOTSET)
    

    for file in args.files:
        update_normer(file)
  
  
            
if __name__ == "__main__":
    main()
