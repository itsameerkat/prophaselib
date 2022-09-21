import os
import sys

import numpy as np
import h5py

from copy import deepcopy
from collections import defaultdict

import extrusionlib as el
# from extrusionlib.lef_factory import *


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

