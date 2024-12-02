import os
import logging
import argparse
from src.predictor import vTrain
from src.config import vTrainConfig

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")

# Base configuration variants
base_variants = [
       "96_512_16", "96_1024_16""96_2048_16", "96_4096_12", "96_8192_16"
]

# Extend variants by appending suffixes
config_variants = []
for variant in base_variants:
    config_variants.extend([
        f"{variant}_8_32_8",
        f"{variant}_4_128_4",
        f"{variant}_8_8_32",
        f"{variant}_16_8_16",
    ])

def process_configs(config_dir, results_dir, folder_name):
    logger.info(f"Processing folder: {folder_name}")

    # Create a subdirectory for results
    folder_results_dir = os.path.join(results_dir, folder_name)
    os.makedirs(folder_results_dir, exist_ok=True)

    for variant in config_variants:
        config_file = os.path.join(config_dir, f"config_{variant}.json")
        result_file = os.path.join(folder_results_dir, f"result_{variant}.txt")
        plot_file = os.path.join(folder_results_dir, f"plot_{variant}.png")
        overlap_file = os.path.join(folder_results_dir, f"overlap_{variant}.txt")

        # Check if the config file exists
        if not os.path.exists(config_file):
            logger.warning(f"Config file not found: {config_file}")
            continue

        logger.info(f"Processing config: {config_file}")
        
        try:
            # Load configuration
            config = vTrainConfig.load_from_file(config_file)

            # Run simulation
            sim = vTrain(config)
            result, breakdown = sim()
            pred_iter_time = max(result.values()) / 1000 / 1000  # Convert to ms
            
            # Save results
            with open(result_file, "w") as f:
                f.write(f"Predicted iteration time: {pred_iter_time:.3f} ms\n")
                f.write("Breakdown:\n")
                for k, v in breakdown.items():
                    f.write(f"{k}: {v}\n")
            
            logger.info(f"Results saved to: {result_file}")
            sim.show_graph(plot_file, overlap_file)

        except Exception as e:
            logger.error(f"Error processing {config_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the results directory")
    args = parser.parse_args()

    # Define folders and process them
    folders = {"llama-96": "config/llama-96"}
    for folder_name, config_dir in folders.items():
        process_configs(config_dir, args.results_dir, folder_name)