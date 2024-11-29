from src.predictor import vTrain
from src.config import vTrainConfig
import os
import argparse
import logging

import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main(args):
    config = vTrainConfig.load_from_file(args.config)

    file_name_ext = (args.config.split('.', 1)[0])[7:]
    folder_results_dir = os.path.join(os.getcwd(), args.results_dir, file_name_ext)
    os.makedirs(folder_results_dir, exist_ok=True)
    logger.info(folder_results_dir)


    sim = vTrain(config)

    result, breakdown = sim()
    pred_iter_time = max(result.values())/1000/1000
    
    logger.info(f"predicted iteration time: {pred_iter_time:.3f} ms")
    variant_name = folder_results_dir[folder_results_dir.rindex('/')+1:]
    with open("%s/%s_result.txt"%(folder_results_dir,variant_name), "w") as f:
        f.write(f"Predicted iteration time: {pred_iter_time:.3f} ms\n")
        f.write("Breakdown:\n")
        for k, v in breakdown.items():
            f.write(f"{k}: {v}\n")
            
    logger.info(f"Results saved")
        
    # except Exception as e:
    #     logger.error(f"Error processing {config_file}: {e}")
    
    sim.show_graph(folder_results_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c, --config", type=str, dest="config")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the results directory")
    args = parser.parse_args()

    main(args)