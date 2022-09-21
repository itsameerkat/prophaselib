import os
import sys

import numpy as np
import h5py

from copy import deepcopy
from collections import defaultdict

import extrusionlib as el
# from extrusionlib.lef_factory import *

def load_cohesin_from_file(cohesin_info, lattice_params):

    chroms, L = lattice_params
    cohesin_loadpath = cohesin_info['loadpath']
    load_cohesin_at = cohesin_info['time_index']

    with h5py.File(f"{cohesin_loadpath}/LEFPositions.h5", mode='r') as f:
        assert f.attrs['system_size'] == L
        assert f.attrs['chroms'] == chroms

        cohesin_positions = (f['/cohesin/positions'][load_cohesin_at,:,:]).astype(float)
        cohesin_positions[cohesin_positions==-1] = np.nan
        try:
            barrier_status = (f['/cohesin/ctcf'][load_cohesin_at,:,:]).astype(bool)
        except:
            print('No CTCF blocking status found in LEFPositions')
            barrier_status = None

        cohesin_attrs = {'d': int(f['cohesin'].attrs['d']),
                         'lambda': int(f['cohesin'].attrs['lambda']),
                         'solution_lifetime': int(f['cohesin'].attrs['solution_lifetime'])
                        }

    return cohesin_attrs, cohesin_positions, barrier_status


def construct_barrier_landscape(barrier_params, lattice_params):

    chroms, L = lattice_params

    with open('/home/sameer/simulations/interphase_washoff/CTCF_landscape_info.json', 'r') as json_file:
        barrier_info = json.load(json_file)
        barrier_positions = np.asarray(barrier_info['positions'])

    assert L==barrier_positions[-1] + 1

    capture_prob = barrier_params['capture_prob']
    release_prob = barrier_params['release_prob']

    capture_dict = {}
    release_dict = {}
    for direction in [-1,+1]:

        dir_dict = {}
        for chrom_start in range(chroms):
            dir_dict.update({(chrom_start*L+pos):capture_prob for pos in barrier_positions})
        capture_dict[direction] = dir_dict

        dir_dict = {}
        for chrom_start in range(chroms):
            dir_dict.update({(chrom_start*L+pos):release_prob for pos in barrier_positions})
        release_dict[direction] = dir_dict

    return capture_dict, release_dict


def calculate_solvent_addition(current_N, current_lattice_lt, current_solvent_lt, new_solvent_lt):

    scale_factor = (current_lattice_lt + new_solvent_lt)/(current_lattice_lt + current_solvent_lt)
    return np.ceil(current_N * scale_factor).astype(int)


def add_cohesins(context, cohesin_inputs, cohesin_load_info, lattice_params, barrier_params, time_factor):

    loaded_attrs, positions, barrier_status = load_cohesin_from_file(cohesin_load_info, lattice_params)


    lam = int(loaded_attrs['lambda'])
    d = int(loaded_attrs['d'])
    lattice_lt = lam/2

    N_old = positions.shape[0]
    old_sol_lt = int(loaded_attrs['solution_lifetime'])
    new_sol_lt = int(cohesin_inputs['solution_lifetime'])

    ## Creating additional Cohesins to solution to account for new lifetime in solvent while keeping d the same
    N_new = calculate_solvent_addition(N_old, lattice_lt, old_sol_lt, new_sol_lt)

    N_cohesin = N_new
    sol_lt = new_sol_lt

    ### Creating CTCF Capture/Release Dict

    capture_dict, release_dict = construct_barrier_landscape(barrier_params, lattice_params)


    ## Creating Cohesin Extruders

    base_template = {'v':1, 'D':0, 'max_v':1, 'max_D':0,
                    'pos':np.nan, 'stalled':False, 'CTCF':False, 'halted':False,
                    }
    base_template.update(cohesin_inputs.get('interaction_info', {}))

    reset_args = ['stalled','pos','halted','CTCF']

    for i in range(N_cohesin):
        legs = []
        sol = True

        if i < positions.shape[0]:
            sol = False

            if np.any(np.isnan(positions[i,:])):
                sol = True


        for direc in [-1, 1]:
            leg_template = deepcopy(base_template)
            leg_template['dir'] = direc

            if i < positions.shape[0]:
                if barrier_status is not None:
                    status = barrier_status[i,(direc+1)//2]
                    leg_template['CTCF'] = status
                    leg_template['halted'] = status

            l = Leg(leg_template=leg_template, reset_args=reset_args)

            if sol:
                l['pos'] = np.nan
            else:
                j = (direc + 1)//2
                l['pos'] = int(positions[i,j])

            l.add_encounter_check('stall', stall_any)
            l.add_cell_check('ctcf_capture', partial(CTCF_capture, capture_probs=capture_dict))
            l.add_cell_check('ctcf_release', partial(CTCF_release, release_probs=release_dict))
            legs.append(l)

        walker = Walker(legs=legs, walker_attrs={'solution':sol, 'name':'cohesin'})
        walker.add_step_check('bind', partial(bind_walker, lifetime= time_factor*sol_lt))
        walker.add_step_check('unbind', partial(unbind, lifetime = time_factor*lattice_lt))

        context.add_walker('cohesin', walker)

    cohesin_params = {
                        'N':N_cohesin,
                        'lambda':lam,
                        'd':d,
                        'solution_lifetime':sol_lt
                     }

    return context, cohesin_params
