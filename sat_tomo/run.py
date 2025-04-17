import os
import subprocess
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run atmospheric tomography programs")
    parser.add_argument("config_file", help="Config file containing program arguments")
    parser.add_argument("--script", default="main.py", help="Target script to run (default: main.py)")
    parser.add_argument("--config_dir", default="config", help="Directory containing configuration files (default: config/)")
    parser.add_argument("--script_dir", default="src", help="Directory containing script file (default: src/)")
    args = parser.parse_args()

    config_file = os.path.join(args.config_dir, args.config_file)
    if not os.path.isfile(config_file):
        print(f"Config file not found: {config_file}")
        sys.exit(1)

    script = os.path.join(args.script_dir, args.script)
    if not os.path.isfile(script):
        print(f"Script not found: {script}")
        sys.exit(1)

    # Load multiple configurations
    with open(config_file) as f:
        all_configs = json.load(f)
    
    if not isinstance(all_configs, list):
        print("Error: JSON must be a list of argument sets")
        sys.exit(1)


    # Run the program for each configuration
    for i, config in enumerate(all_configs, start=1):
        print(f"\n\n=== Running configuration {i} ===")
        
        cmd_args = []
        for key, value in config.items():
            d = "--"
            if len(key) == 1:
                d = "-"
            
            if isinstance(value, bool):
                if value:
                    cmd_args.append(f"{d}{key}")
            elif isinstance(value, list):
                # Add each item in the list as a separate argument
                cmd_args.append(f"{d}{key}")
                cmd_args.extend(map(str, value))
            else:
                cmd_args.extend([f"{d}{key}", str(value)])

        cmd = ["python", script] + cmd_args
        # print(cmd)
        subprocess.run(cmd)

if __name__ == "__main__":
    main()