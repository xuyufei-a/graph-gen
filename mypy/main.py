# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import utils
import argparse
from configs.datasets_config import qm9_with_h, qm9_without_h
from qm9 import dataset
from qm9.models import get_model, DistributionNodes

from equivariant_diffusion.utils import assert_correctly_masked
import torch
import pickle
import qm9.visualizer as vis
from qm9.analyze import check_stability
from os.path import join
from qm9.sampling import sample_chain, sample
from configs.datasets_config import get_dataset_info
import os

def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)
# 

def save_and_sample_chain(args, eval_args, device, flow,
                          n_tries, n_nodes, dataset_info, id_from=0,
                          num_chains=100):
    for i in range(num_chains):
        target_path = f'eval/chain_{i}/'

        one_hot, charges, x = sample_chain(
            args, device, flow, n_tries, dataset_info)

        vis.save_xyz_file(
            join(eval_args.model_path, target_path), one_hot, charges, x,
            dataset_info, id_from, name='chain')

        vis.visualize_chain_uncertainty(
            join(eval_args.model_path, target_path), dataset_info,
            spheres_3d=True)

    return one_hot, charges, x

def save_point_file(path, one_hot, charges, positions, dataset_info, id_from=0, name='molecule', node_mask=None, dims_mask=None):
    try:
        os.makedirs(path)
    except OSError:
        pass

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)
    n_dim = dims_mask.sum(dim=1)
    
    for batch_i in range(one_hot.size(0)):
        f = open(path + name + '_' + "%03d.txt" % (batch_i + id_from), "w")
        f.write("%d %d\n\n" % (atomsxmol[batch_i], int(n_dim[batch_i])))
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        n_atoms = int(atomsxmol[batch_i])
       
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = dataset_info['atom_decoder'][atom]

            
            f.write(f'{atom}')
            for j in range(int(n_dim[batch_i])):
                f.write(f' {positions[batch_i, atom_i, j]:.9f}')
            f.write('\n')
        f.close()

# TODO: my function
def sample_different_sizes_and_dims(args, eval_args, device, generative_model,
                                    nodes_dist, dataset_info, n_samples=10, dims_dist=None):
    # nodes_list: qm9.models.DistributionNodes

    nodesxsample = nodes_dist.sample(n_samples)
    dimsxsample = dims_dist.sample(n_samples)

    dims_mask = torch.zeros((len(nodesxsample), eval_args.max_n_dims))
    for i in range(len(dimsxsample)):
        dims_mask[i, 0:dimsxsample[i]] = 1
    
    one_hot, charges, x, node_mask = sample(
        args, device, generative_model, dataset_info,
        nodesxsample=nodesxsample)

    save_point_file(
        join(eval_args.model_path, 'eval/molecules/'), one_hot, charges, x,
        id_from=0, name='molecule', dataset_info=dataset_info,
        node_mask=node_mask, dims_mask=dims_mask)
 

def sample_different_sizes_and_save(args, eval_args, device, generative_model,
                                    nodes_dist, dataset_info, n_samples=10):
    # nodes_list: qm9.models.DistributionNodes

    nodesxsample = nodes_dist.sample(n_samples)
    
    one_hot, charges, x, node_mask = sample(
        args, device, generative_model, dataset_info,
        nodesxsample=nodesxsample)

    vis.save_xyz_file(
        join(eval_args.model_path, 'eval/molecules/'), one_hot, charges, x,
        id_from=0, name='molecule', dataset_info=dataset_info,
        node_mask=node_mask)


def sample_only_stable_different_sizes_and_save(
        args, eval_args, device, flow, nodes_dist,
        dataset_info, n_samples=10, n_tries=50):
    assert n_tries > n_samples

    nodesxsample = nodes_dist.sample(n_tries)
    one_hot, charges, x, node_mask = sample(
        args, device, flow, dataset_info,
        nodesxsample=nodesxsample)

    counter = 0
    for i in range(n_tries):
        num_atoms = int(node_mask[i:i+1].sum().item())
        atom_type = one_hot[i:i+1, :num_atoms].argmax(2).squeeze(0).cpu().detach().numpy()
        x_squeeze = x[i:i+1, :num_atoms].squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

        num_remaining_attempts = n_tries - i - 1
        num_remaining_samples = n_samples - counter

        if mol_stable or num_remaining_attempts <= num_remaining_samples:
            if mol_stable:
                print('Found stable mol.')
            vis.save_xyz_file(
                join(eval_args.model_path, 'eval/molecules/'),
                one_hot[i:i+1], charges[i:i+1], x[i:i+1],
                id_from=counter, name='molecule_stable',
                dataset_info=dataset_info,
                node_mask=node_mask[i:i+1])
            counter += 1

            if counter >= n_samples:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument(
        '--n_tries', type=int, default=10,
        help='N tries to find stable molecule for gif animation')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='number of atoms in molecule for gif animation')

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    # TODO 
    eval_args.max_n_dims = 18
    flow, nodes_dist, prop_dist = get_model(
        args, device, dataset_info, dataloaders['train'], n_dims=eval_args.max_n_dims)
    flow.to(device)

    #TODO 
    # for name, p in flow.named_parameters():
    #     print(name, p.shape)
    # exit()

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn),
                                 map_location=device)

    flow.load_state_dict(flow_state_dict)

    # histogram = {i:1 for i in range(1, eval_args.max_n_dims + 1)}
    histogram = dataset_info['ranks'] 
    dims_dist = DistributionNodes(histogram)
    sample_different_sizes_and_dims(
        args, eval_args, device, flow, nodes_dist,
        dataset_info=dataset_info, n_samples=1, dims_dist=dims_dist)
    print('finished')

if __name__ == "__main__":
    main()