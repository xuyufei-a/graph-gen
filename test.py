# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from qm9 import dataset
from qm9.models import get_model, DistributionNodes
import os
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import sample
from qm9.analyze import analyze_stability_for_molecules, analyze_node_distribution
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
import qm9.losses as losses
from mypy.utils.MoleculeMetrics import MoleculeMetrics
from mypy.utils.molecule_transform import inverse_SRD

try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)

# TODO: my function
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
                                    nodes_dist, dims_dist, dataset_info, n_samples=10):
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
    
    return one_hot, x, node_mask, dims_mask

def analyze_and_save(args, eval_args, device, generative_model,
                     nodes_dist, dims_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = []

    start_time = time.time()
    for i in range(int(n_samples/batch_size)):
        one_hot, x, node_mask, dims_mask =  sample_different_sizes_and_dims(args, eval_args, device, generative_model
                                        , nodes_dist, dims_dist, dataset_info, batch_size)

        node_nums = node_mask.sum(dim=1)
        dim_nums = dims_mask.sum(dim=1)

        for j in range(batch_size):
            node_num = int(node_nums[j].item())
            dim_num = int(dim_nums[j].item())
            tmp = torch.zeros_like(x[j])
            tmp[0:node_num, 0:dim_num] = 1
            x[j] = x[j] * tmp

        adjs = inverse_SRD(x)
        for j in range(batch_size):
            node_num = int(node_nums[j].item())
            dim_num = int(dim_nums[j].item())
            atom_type = torch.argmax(one_hot[j], dim=1)[0:node_num]
            adj = adjs[j]
            molecules.append((adj, atom_type)) 

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        metric = MoleculeMetrics()
        vun = metric.evaluate(molecules)
        return vun


def test(args, flow_dp, nodes_dist, device, dtype, loader, partition='Test', num_passes=1):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0
    for pass_number in range(num_passes):
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get data
                x = data['positions'].to(device, dtype)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

                batch_size = x.size(0)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                h = {'categorical': one_hot, 'integer': charges}

                if len(args.conditioning) > 0:
                    context = prepare_context(args.conditioning, data).to(device, dtype)
                    assert_correctly_masked(context, node_mask)
                else:
                    context = None

                # transform batch through flow
                nll, _, _ = losses.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask,
                                                        edge_mask, context)
                # standard nll from forward KL

                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')

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

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    eval_args.max_n_dims = 18
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'], n_dims=eval_args.max_n_dims)
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)

    # Analyze validity, uniqueness and novelty
    histogram = dataset_info['ranks'] 
    dims_dist = DistributionNodes(histogram)
    rdkit_metrics = analyze_and_save(
        args, eval_args, device, generative_model, nodes_dist, dims_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen)

    if rdkit_metrics is not None:
        rdkit_metrics = rdkit_metrics[0]
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
    else:
        print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")


    # # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
    # if args.dataset == 'geom':
    #     val_name = 'val'
    #     num_passes = 1
    # else:
    #     val_name = 'valid'
    #     num_passes = 5

    # TODO
    # # Evaluate negative log-likelihood for the validation and test partitions
    # val_nll = test(args, generative_model, nodes_dist, device, dtype,
    #                dataloaders[val_name],
    #                partition='Val')
    # print(f'Final val nll {val_nll}')
    # test_nll = test(args, generative_model, nodes_dist, device, dtype,
    #                 dataloaders['test'],
    #                 partition='Test', num_passes=num_passes)
    # print(f'Final test nll {test_nll}')

    # print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict)
    # with open(join(eval_args.model_path, 'eval_log.txt'), 'w') as f:
    #     print(f'Overview: val nll {val_nll} test nll {test_nll}',
    #           stability_dict,
    #           file=f)


if __name__ == "__main__":
    main()