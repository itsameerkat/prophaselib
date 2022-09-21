

def calculate_titration(N, lattice_lt, current_solvent_lt, new_solvent_lt):

    scale_factor = (lattice_lt + new_solvent_lt)/(lattice_lt + current_solvent_lt)
    return round(N * scale_factor)


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
