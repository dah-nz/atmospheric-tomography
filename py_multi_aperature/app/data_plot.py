import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import util
import strehl
from progressbar import progressbar

file_name = "TEST.DATA"

old = False
if old:
    target_section = "Index,Target Position Azimuth [rad],Target Position Theta [rad]"
    asterism_section = "Index,Asterism [rad]"
    mode_section = "Index,Target Modes"
    output_section = "Target Position Index,Asterism Index,Target Mode Index,Tomography Output"
else:
    target_section = "TEST_TARGET_POSITIONS"
    asterism_section = "TEST_ASTERISMS"
    mode_section = "TEST_TARGET_MODES"
    output_section = "TEST_OUTPUTS"

targets = {}
asterisms = {}
modes = {}
output = {}
output_rmse = {}
output_strehl = {}

def read_target(line):
    data = literal_eval(line)
    if old:
        targets[data[0]] = (data[1], data[2])
    else:
        targets[data[0]] = data[1]

def read_asterism(line):
    data = literal_eval(line)
    asterisms[data[0]] = (data[1], data[2])

def read_mode(line):
    try:
        data = literal_eval(line)
        if data[0] != 400:
            modes[data[0]] = data[1]
    except ValueError:
        print(line)

def read_output(line):
    data = literal_eval(line)
    key = (data[0], data[1], data[2])
    if data[2] != 400:
        output[key] = data[3]

def rmse(expected, actual):
    e = expected - actual
    se = e * e
    mse = np.mean(se)
    return np.sqrt(mse)

def main():
    section = ""
    with open(file_name, 'r') as f:
        for line in progressbar(f.readlines()):
            line = line.strip()
            if line == target_section:
                section = "targets"
            elif line == asterism_section:
                section = "asterisms"
            elif line == mode_section:
                section = 'modes'
            elif line == output_section:
                section = 'output'
            else:
                if section == "targets":
                    read_target(line)
                if section == "asterisms":
                    read_asterism(line)
                if section == "modes":
                    read_mode(line)
                if section == "output":
                    read_output(line)

    # calculate RMSE for each output
    for (t, a, m), data in progressbar(output.items()):
        expected = np.sum(modes[m], axis=0)
        output_rmse[(t,a,m)] = rmse(expected, [0.0] + data)
        output_strehl[(t,a,m)] = ((strehl.strehl_from_modes([0.0] + data) / strehl.strehl_from_modes(expected)) - 1) * 100

    # error = [i for _, i in output_rmse.items()]
    # plt.plot(error)
    # plt.show()

    # plot average error for different target angles
    def get_target_angle_results(data):
        pos_data = {}
        pos_count = {}
        for k, err in data.items():
            pos = targets[k[0]]
            a = util.rad2arcsec(pos[0])
            try:
                pos_data[a] += err
                pos_count[a] += 1
            except KeyError:
                pos_data[a] = err
                pos_count[a] = 1
        
        for k in pos_count.keys():
            pos_data[k] /= pos_count[k]
        return pos_data

    pos_data = get_target_angle_results(output_rmse)
    
    plt.figure()
    plt.title("RMS Error based on target angle from azimuth")
    plt.xlabel("Angle from Azimuth (arcseconds)")
    plt.ylabel("Average RMSE of Tomography")
    plt.plot(pos_data.keys(), pos_data.values())

    pos_data = get_target_angle_results(output_strehl)

    plt.figure()
    plt.title("Strehl Ratio change based on target angle from azimuth")
    plt.xlabel("Angle from Azimuth (arcseconds)")
    plt.ylabel("Average Change in Strehl Ratio (%)")
    plt.plot(pos_data.keys(), pos_data.values())


    # plot average error for different target positions
    def get_position_results(data):
        pos_data = {}
        pos_count = {}
        for k, err in data.items():
            pos = targets[k[0]]
            a = pos[1]
            try:
                pos_data[a] += err
                pos_count[a] += 1
            except KeyError:
                pos_data[a] = err
                pos_count[a] = 1
        
        for k in pos_count.keys():
            pos_data[k] /= pos_count[k]
        return pos_data

    pos_data = get_position_results(output_rmse)
    
    plt.figure()
    plt.title("RMS Error based on target location")
    plt.xlabel("Angle around centre (radians) [0 = X only, pi/2 = Y only]")
    plt.ylabel("Average RMSE of Tomography")
    plt.plot(pos_data.keys(), pos_data.values())

    pos_data = get_position_results(output_strehl)

    plt.figure()
    plt.title("Strehl Ratio Change based on target location")
    plt.xlabel("Angle around centre (radians) [0 = X only, pi/2 = Y only]")
    plt.ylabel("Average Change in Strehl Ratio (%)")
    plt.plot(pos_data.keys(), pos_data.values())

    asterism_types = ['equal', 'random_single', 'full_random'] 
    # plot average error for different asterisms
    def get_asterism_results(data):
        def setup(d):
            for t in asterism_types:
                d[t] = {}
            return d
        ast_rad_data = setup({})
        ast_rad_count = setup({})
        ast_num_data = setup({})
        ast_num_count = setup({})
        for k, err in data.items():
            a = asterisms[k[1]]
            ast_type = a[0]
            ast_pos = a[1]
            radius = util.rad2arcsec(ast_pos[0][0])
            num = len(ast_pos)
            try:
                ast_rad_data[ast_type][radius] += err
                ast_rad_count[ast_type][radius] += 1
            except KeyError:
                ast_rad_data[ast_type][radius] = err
                ast_rad_count[ast_type][radius] = 1

            try:
                ast_num_data[ast_type][num] += err
                ast_num_count[ast_type][num] += 1
            except KeyError:
                ast_num_data[ast_type][num] = err
                ast_num_count[ast_type][num] = 1
        
        for k in ast_rad_count.keys():
            for k2 in ast_rad_count[k].keys():
                ast_rad_data[k][k2] /= ast_rad_count[k][k2]

        for k in ast_num_count.keys():
            for k2 in ast_num_count[k].keys():
                ast_num_data[k][k2] /= ast_num_count[k][k2]
        
        return ast_rad_data, ast_num_data
    
    ast_rad_data, ast_num_data = get_asterism_results(output_rmse)

    plt.figure()
    plt.title("RMS Error based on asterism radius")
    plt.xlabel("Asterism Radius [arcseconds]")
    plt.ylabel("Average RMSE of Tomography")
    for t in asterism_types:
        plt.plot(ast_rad_data[t].keys(), ast_rad_data[t].values(), label=t)
    plt.legend()

    plt.figure()
    plt.title("RMS Error based on asterism star count")
    plt.xlabel("Number of Stars")
    plt.ylabel("Average RMSE of Tomography")
    for t in asterism_types:
        plt.plot(ast_num_data[t].keys(), ast_num_data[t].values(), label=t)
    plt.legend()

    ast_rad_data, ast_num_data = get_asterism_results(output_strehl)

    plt.figure()
    plt.title("Strehl Ratio change based on asterism radius")
    plt.xlabel("Asterism Radius [arcseconds]")
    plt.ylabel("Average Change in Strehl Ratio (%)")
    for t in asterism_types:
        plt.plot(ast_rad_data[t].keys(), ast_rad_data[t].values(), label=t)
    plt.legend()

    plt.figure()
    plt.title("Strehl Ratio change based on asterism star count")
    plt.xlabel("Number of Stars")
    plt.ylabel("Average Change in Strehl Ratio (%)")
    for t in asterism_types:
        plt.plot(ast_num_data[t].keys(), ast_num_data[t].values(), label=t)
    plt.legend()

    # plot average error for different mode combinations
    def get_mode_results(data):
        mode_data = {}
        mode_count = {}
        for k, err in data.items():
            pos = modes[k[2]]
            try:
                mode_data[k[2]] += err
                mode_count[k[2]] += 1
            except KeyError:
                mode_data[k[2]] = err
                mode_count[k[2]] = 1
        
        for k in mode_count.keys():
            mode_data[k] /= mode_count[k]
        return mode_data

    mode_data = get_mode_results(output_rmse)
    
    plt.figure()
    plt.title("RMS Error based on target modes")
    plt.xlabel("Mode Combination Index")
    plt.ylabel("Average RMSE of Tomography")
    md = dict(sorted(mode_data.items()))
    plt.plot(md.keys(), md.values())

    mode_data = get_mode_results(output_strehl)

    plt.figure()
    plt.title("Strehl Ratio change based on target modes")
    plt.xlabel("Mode Combination Index")
    plt.ylabel("Average Change in Strehl Ratio (%)")
    md = dict(sorted(mode_data.items()))
    plt.plot(md.keys(), md.values())

    plt.show()





if __name__ == "__main__":
    main()