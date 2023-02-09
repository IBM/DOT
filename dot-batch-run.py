import os
import sys
import json
from datetime import datetime
import subprocess
from tqdm import tqdm
import time


parallel = True
    
problem = {
    # paralelepiped enclosing the phantom 
    "lb_x": 0.0,
    "lb_y": 0.0,
    "lb_z": 0.0,

    "rb_x": 1.3,  # cm
    "rb_y": 1.3,  # cm
    "rb_z": 1.4,  # cm

    # phantom support
    "phantom_lb_x": 0.25,
    "phantom_lb_y": 0.25,
    "phantom_lb_z": 0.2,

    "phantom_rb_x": 1.05,
    "phantom_rb_y": 1.05,
    "phantom_rb_z": 1.1,
    
    # observations
    "data_folder": os.path.expanduser('~')+'/Library/CloudStorage/Box-Box/UCD_laser_source_videos/All_Phantom_Data/ASCIIs',
    "source_file": '/NewASCIIsSource/RBsource',
    "measurements_top_file": '/medFilt_RED3mmTop7uM',
    "measurements_side_file": '/RB3mmSide7uM'
}

solver_params = {
    "name": "born",
    "results_folder": "results",

    # observed faces, either "top" or "top_and_sides"
    "observed_faces": "top_and_sides",
    
    # number of elements
    "Nel_x": 22,
    "Nel_y": 22,
    "Nel_z": 24,
    # polynomial degree of elements
    "degree": 1,
    
    # number of elements for a boundary layer width
    "bound_num_el": 3,
    # Define supports for local Total Variation regularization
    "Nc_x": 16,
    "Nc_y": 16,
    "Nc_z": 16,
    
    "linear_solver": 'mumps',
    # OSQP, SCS, ECOS, MOSEK
    "opt_method": "OSQP",
    # so far possible options: 0, 1, 2, 3
    "variant": 3,
    "init_test": False,
    "max_iter": 50000,
    "step_diff": 0.01,
    "max_steps": 10,
    
    # type of mask: 0 (no mask), 1 (data driven mask), 2 (ground truth mask)
    "mask": 0
}


updates = [
    {"name": "hybrid", "opt_method": "SCS", "variant": 0, "mask": 0},
    {"name": "hybrid", "opt_method": "SCS", "variant": 1, "mask": 0},
    {"name": "hybrid", "opt_method": "SCS", "variant": 2, "mask": 0},
    {"name": "hybrid", "opt_method": "SCS", "variant": 3, "mask": 0},
    
    {"name": "born", "opt_method": "SCS", "variant": 0, "mask": 0},
    {"name": "born", "opt_method": "SCS", "variant": 1, "mask": 0},
    {"name": "born", "opt_method": "SCS", "variant": 2, "mask": 0},
    {"name": "born", "opt_method": "SCS", "variant": 3, "mask": 0},

    {"name": "hybrid", "variant": 0, "mask": 0},
    {"name": "hybrid", "variant": 1, "mask": 0},
    {"name": "hybrid", "variant": 2, "mask": 0},
    {"name": "hybrid", "variant": 3, "mask": 0},
    
    {"name": "born", "variant": 0, "mask": 0},
    {"name": "born", "variant": 1, "mask": 0},
    {"name": "born", "variant": 2, "mask": 0},
    {"name": "born", "variant": 3, "mask": 0}
]

procs = []

for update in tqdm(updates):
    params = solver_params.copy()
    for key in update:
        params[key]=update[key]
        
    # create output directory and save basic config
    time_stamp = datetime.now().strftime('%b_%d_%H')
    name = params['name']+str(params['variant'])+'_'+params['opt_method']+\
           '_mask'+str(params['mask'])+'_'+params['observed_faces']
    dir_name = name+'_'+time_stamp
    
    params["results_folder"] = dir_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(dir_name+'/problem.json', 'w') as fp:
        json.dump(problem, fp)
    with open(dir_name+'/solver_params.json', 'w') as fp:
        json.dump(params, fp)
        
    
    # run optimisation in a subprocess
    try:
        bashCommand = f'python dot-experiment.py {dir_name}'    
        with open(f'{dir_name}/{name}.log', 'w') as output_fh:
            if parallel:
                p = subprocess.Popen(bashCommand.split(), stdout=output_fh, stderr=output_fh)
                procs.append(p)
            else:
                process = subprocess.check_call(bashCommand.split(), stdout=output_fh, stderr=output_fh)
                # process = subprocess.run(bashCommand.split(), stdout=output_fh, stderr=output_fh)                            
    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print("ERROR: An exception occurred while running an experiment. For details, Check log file in directory "+dir_name)
        print(e.output)
    except:
        print("An unknown exception occurred")
        # print(CalledProcessError.output)

if parallel:
    cont = True
    while cont:
        finished = 0
        for p in procs:            
            if p.poll() is not None:
                finished = finished + 1
        if finished == len(procs):
            cont = False
            print(f'Finished {finished} experiments out of {len(procs)}.')
            print('All experiments finished.')
        else:
            print(f'Finished {finished} experiments out of {len(procs)}.')
            time.sleep(30)
