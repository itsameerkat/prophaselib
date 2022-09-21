import os
import sys
import time
import glob
import numpy as np
import h5py
import json
from functools import partial
from copy import deepcopy
from collections import defaultdict

import simtk.openmm 
import polychrom
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI

import extrusionlib as el
from extrusionlib.lef_factory import *
from extrusionlib.bond_propagation import extrusionPropagator


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


def add_condensins(context, condensin_inputs, lattice_params): 
        
    chroms, L = lattice_params
    
    lam = condensin_inputs['lambda']
    d = condensin_inputs['d']
    sol_lt = condensin_inputs['solution_lifetime']
    v = condensin_inputs['velocity']
    
    if np.isfinite(lam):
        lattice_lt = lam/2
        f = lattice_lt/(lattice_lt + sol_lt)
        N_condensin = int(chroms*L/(d * f))
    else:
        N_condensin = int(chroms*L/d)
    
    
    ## Determining collision rule
    
    collision_type = condensin_inputs.get('interaction_info', 'stall')
    
    if collision_type == 'unload':
        coll_fn = partial(unbind_other, lifetime=1)
        
    elif collision_type == 'push':
        coll_fn = partial(push_thru_CTCF, lifetime=1)
        
    elif collision_type == 'hop bypass':
        coll_fn = partial(hop_bypass, lifetime=1)
        
    elif collision_type == 'swap bypass':
        coll_fn = partial(swap_bypass, lifetime=1)

    elif collision_type == 'stall':
        coll_fn = stall_any
        
    else:
        raise
        
        
        
    ## Creating Condensin Extruders
        
    base_template = {'v':v, 'D':0, 'max_v':v, 'max_D':0, 'pos':np.nan, 'stalled':False, 'halted':False} 
    reset_args = ['stalled','pos','halted']

    for _ in range(N_condensin):
        legs = []
        for direc in [-1, 1]: 
            leg_template = deepcopy(base_template)
            leg_template['dir'] = direc 
            l = Leg(leg_template=leg_template, reset_args=reset_args)
            l.add_encounter_check(collision_type, coll_fn)
            legs.append(l)

        walker = Walker(legs=legs, walker_attrs={'solution':True, 'name':'condensin'})
        walker.add_step_check('bind', partial(bind_walker, lifetime=sol_lt))
        if np.isfinite(lam):
            walker.add_step_check('unbind', partial(unbind, lifetime=lattice_lt))

        context.add_walker('condensin', walker)
    
    condensin_params = {
                        'N':N_condensin, 
                        'lambda':lam, 
                        'd':d, 
                        'solution_lifetime':sol_lt,
                        'velocity':v,
                       }
    
    return context, condensin_params


def create_savefile(folder, extruder_dict, lattice_params, T):
    
    chroms, L = lattice_params
    
    with h5py.File(f'{folder}/LEFPositions.h5', mode='w') as f:

        f.attrs['system_size'] = L
        f.attrs['chroms'] = chroms
        f.attrs['time'] = T
        
        for extruder_name in extruder_dict.keys():
            extruder_info = extruder_dict[extruder_name]
            N_ext = extruder_info['N']
            
            g = f.create_group(extruder_name)
            g.attrs['lambda'] = extruder_info['lambda']
            g.attrs['d'] = extruder_info['d']
            g.attrs['solution_lifetime'] = extruder_info['solution_lifetime']
            if extruder_name == 'condensin':
                g.attrs['velocity'] = extruder_info['velocity']
                
            g.create_dataset('positions', 
                             shape=(T, N_ext, 2), 
                             dtype=np.int32, 
                             compression="gzip")
            
            if extruder_name == 'cohesin':
                g.create_dataset('ctcf', 
                                 shape=(T, N_ext, 2), 
                                 dtype=bool, 
                                 compression="gzip")

        
        
def run_extrusion(context, T, lattice_params, extruder_dict, savefolder):
    
    create_savefile(savefolder, extruder_dict, lattice_params, T)
    
    steps = 50
    bins = np.linspace(0, T, steps, dtype=int)

    position_obj = {}
    with h5py.File(f'{savefolder}/LEFPositions.h5', mode='a') as f:

        for extruder_name in extruder_dict.keys():
            g = f[extruder_name]

            position_obj[extruder_name] = g['positions']
            
            if extruder_name == 'cohesin':
                ctcf_obj = g['ctcf']
            
        for st,end in zip(bins[:-1], bins[1:]):
            
            positions = defaultdict(list)
            ctcf_status = []
            for i in range(st, end):
                if i % 1000 == 0:
                    print(f'{i} of {T} steps taken')
                    
                context.step()

                for extruder_name in extruder_dict.keys():
                    mask = (context.walker_types==extruder_name)
                    
                    pos = np.vstack(tuple([lef['pos',:] for lef in context.walkers[mask]]))
                    positions[extruder_name].append(pos)
                    
                    if extruder_name == 'cohesin':
                        ctcf = np.vstack(tuple([lef['CTCF',:] for lef in context.walkers[mask]]))
                        ctcf_status.append(ctcf)
                
            for extruder_name in extruder_dict.keys():
                pos = np.array(positions[extruder_name])
                pos[np.isnan(pos)] = -1
                position_obj[extruder_name][st:end] = pos.astype(int)

                if extruder_name == 'cohesin':
                    ctcf = np.array(ctcf_status).astype(bool)
                    ctcf_obj[st:end] = ctcf


def unbind_other(leg, other, context, lifetime):
    walker = other['walker']

    if not other['unloadable']:
        leg['stall'] = True
        return    
 
    if np.random.random() < (1/lifetime):
        context[walker['pos',:]] = 0
        walker.reset()
    else:
        leg['stall'] = True
        return
    
def swap_bypass(leg, other, context, lifetime):
    step_dir = other['pos'] - leg['pos']
    active_dir = leg['dir']
    
    if step_dir != active_dir:
        return 
    
    if not other['passable']:
        leg['stall'] = True
        return
    
    if np.random.random() < (1/lifetime):
        p1 = leg['pos']
        p2 = other['pos']
        leg['pos'] = p2
        other['pos'] = p1
        context[p1] = other
        context[p2] = leg
    else:
        leg['stall'] = True
        return
    
    
def hop_bypass(leg, other, context, lifetime):
    step_dir = other['pos'] - leg['pos']
    active_dir = leg['dir']
    
    if step_dir != active_dir:
        return 
    
    if (not other['passable']):
        leg['stall'] = True
        return
    
    if np.random.random() < (1/lifetime):
        last_leg = other
        while True:
            new = context[last_leg['pos']+active_dir]
            if isinstance(new, Leg):
                if not new['passable']:
                    leg['stall'] = True
                    return
                
                last_leg = new
                
            elif not new:
                break
                
            else:
                leg['stall'] = True
                return
                
        p1 = leg['pos']
        p2 = last_leg['pos'] + leg['dir']
        leg['pos'] = p2
        
        context[p1] = 0
        context[p2] = leg
        
    else:
        
        leg['stall'] = True
        return
    
    
def push_thru_CTCF(leg, other, context, lifetime):
    """
    Encounter action. Pushes other legs if they are pushable. 
    Is only performed when moving in the direction of motion, and with probability 1/lifetime. 
    (set lifetime=1 to be performed always). 
        
    """
    step_dir = other['pos'] - leg['pos']
    active_dir = leg['dir']
    
    if step_dir != active_dir:
        return # The step is diffusive and I don't want to push on a diffusive step
    
    if np.random.random() < (1/lifetime):
        if other['pushable']:
            leg_list = [other]
            while True:
                new = context[leg_list[-1]['pos']+active_dir]
                if isinstance(new, Leg):
                    if not new['pushable']:
                        leg['stall'] = True
                        return
                    
                    leg_list.append(new)
                elif not new:
                    break
                else:
                    leg['stall'] = True
                    return

            for l in leg_list[::-1]:
                context[l['pos']] = 0
                l['pos'] = l['pos']+active_dir
                context[l['pos']] = l

            context[leg['pos']] = 0 
            leg['pos'] = leg['pos']+active_dir
            context[leg['pos']] = leg

        else:
            leg['stall'] = True
    else:
        leg['stall'] = True
        return
    
    

def do_polymer_sim(savefolder, polymer_params, init_conformation, gpu, iter_print=''):
    
    N_monomers = polymer_params['N_monomers']
    chroms = polymer_params['chroms']

    
    density = polymer_params['density'] 
    col_rate = polymer_params['col_rate']
    Nmd = polymer_params['Nmd']
    
    box = (chroms*N_monomers / density) ** 0.33
    conformation = init_conformation
    
    # Simulation Info
    smcBondWiggleDist = 0.2
    smcBondDist = 1.0
    
    f =  h5py.File(f"{savefolder}/LEFPositions.h5", mode='r')
    LEF_positions = [f[f'{name}/positions'] for name in f.keys()]        
    milker = extrusionPropagator(LEF_positions)
    
    reporter = HDF5Reporter(folder=savefolder, max_data_length=100, overwrite=False, blocks_only=False)

    T_steps = polymer_params['T_steps']
    
#     for integrations_per_restart in [100, 50, 30]:
#         if (T_steps % integrations_per_restart) == 0:
#             break
            
#             integrations_per_restart = 50
    
    
    
    integrations_per_save = polymer_params['ints_per_save']
    integrations_per_restart = 5*integrations_per_save
    
    assert (integrations_per_restart % integrations_per_save) == 0
    assert (T_steps % integrations_per_restart) == 0 
    
    num_inits  = T_steps // integrations_per_restart
    for iteration in range(num_inits):
        print(iter_print)
        print(f'*******Iteration No. {iteration+1} of {num_inits}**********\n\n')

        a = Simulation(
                platform="cuda",
                integrator="variableLangevin", 
                error_tol=0.01, 
                GPU = gpu, 
                collision_rate=col_rate, 
                N = len(conformation),
                PBCbox=(box, box, box),
                reporters=[reporter],
                precision="single")  

        a.set_data(conformation, center=True)
        print('\n')

        a.add_force(
            forcekits.polymer_chains(a,

                            chains=[(i*N_monomers, (i+1)*N_monomers, False) for i in range(chroms)],
                            bond_force_func=forces.harmonic_bonds,
                            bond_force_kwargs={
                                                'bondLength':1.0,
                                                'bondWiggleDistance':0.1, 
                                              },

                            angle_force_func=forces.angle_force,
                            angle_force_kwargs={
                                                 'k':1.5  
                                               },

                            nonbonded_force_func=forces.polynomial_repulsive,
                            nonbonded_force_kwargs={
                                                 'trunc':1.5,
                                                 'radiusMult':1.05,
                                               },
                                    except_bonds=True,

                                    )
                   )
        print('\n')

        # copied from addBond
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        bondDist = smcBondDist * a.length_scale

        activeParams = {"length":bondDist,"k":kbond}
        inactiveParams = {"length":bondDist, "k":0}
        milker.setParams(activeParams, inactiveParams)

        # this step actually puts all bonds in and sets first bonds to be what they should be
        milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                    blocks=integrations_per_restart
                    )

        # If your simulation does not start, consider using energy minimization below
        if iteration==0:
            a.local_energy_minimization() 
        else:
            a._apply_forces()
        print('\n')

        for i in range(integrations_per_restart):        
            if i % integrations_per_save == (integrations_per_save - 1):  
                a.do_block(steps=int(Nmd))
            else:
                a.integrator.step(int(Nmd))  # do steps without getting the positions from the GPU (faster)
            if i < integrations_per_restart - 1: 
                curBonds, pastBonds = milker.step(a.context)  # this updates bonds. You can do something with bonds here
        conformation = a.get_data()  # save data and step, and delete the simulation
        del a

        reporter.blocks_only = True  # Write output hdf5-files only for blocks

        time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

    reporter.dump_data()
    f.close()
    

def acquire_ctcf_status(cohesin_inputs, cohesin_load_info, barrier_params, lattice_params, T, savefolder): 
    
    chroms, L = lattice_params
    context = el.lef_dynamics.Context(chroms*[L])

    context, cohesin_params =  add_cohesins(context, 
                                            cohesin_inputs, cohesin_load_info, lattice_params, 
                                            barrier_params,  1
                                           ) 
    
    
    run_extrusion(context, T, lattice_params, {'cohesin':cohesin_params}, savefolder)
    
    
def prophase_extrusion(condensin_inputs, cohesin_inputs, cohesin_load_info, barrier_params, lattice_params, T, savefolder): 
    
    chroms, L = lattice_params
    context = el.lef_dynamics.Context(chroms*[L])
    
    
    time_scale_factor = condensin_inputs['velocity']
    context, cohesin_params =  add_cohesins(context, 
                                            cohesin_inputs, cohesin_load_info, 
                                            lattice_params, barrier_params,  
                                            time_scale_factor
                                           ) 
    
    context, condensin_params = add_condensins(context, condensin_inputs, lattice_params)
    
    run_extrusion(context, T, lattice_params, {'cohesin':cohesin_params,'condensin':condensin_params}, savefolder)