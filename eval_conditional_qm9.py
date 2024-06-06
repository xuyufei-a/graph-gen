import argparse
from os.path import join
import torch
import pickle
from qm9.models import get_model, DistributionNodes
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.sampling import sample
from qm9.property_prediction.main_qm9_prop import test
from qm9.property_prediction import main_qm9_prop
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import qm9.visualizer as vis
from mypy.utils.molecule_transform import srd_to_smiles, smile_to_xyz
from mypy.utils.check import check_mask
from mypy.utils.molecule_transform import smile_to_xyz
from mypy.tmp_script.get_split import get_splits


def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier


def get_args_gen(dir_path):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_gen = pickle.load(f)

    # TODO: unknown use
    # assert args_gen.dataset == 'qm9_second_half'

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'
    return args_gen


def get_generator(dir_path, dataloaders, device, args_gen, property_norms):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)

    # TODO modify n_dims
    model, nodes_dist, prop_dist = get_model(args_gen, device, dataset_info, dataloaders['train'], n_dims=dataset_info['max_n_dims'])
    fn = 'generative_model_ema.npy' if args_gen.ema_decay > 0 else 'generative_model.npy'
    model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
    model.load_state_dict(model_state_dict)

    # The following function be computes the normalization parameters using the 'valid' partition

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), nodes_dist, prop_dist, dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders

# import tarfile
# class QM9_dataloader:
#     def __init__(self, args):
#         self.device = args.device
#         file_idx_list = get_splits()['test']
#         # print(file_idx_list)
#         self.tot_num = len(file_idx_list)
#         self.i = 0
#         atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
#         self.positions = torch.zeros((self.tot_num, 29, 3), device=args.device)
#         self.one_hots = torch.zeros((self.tot_num, 29, 5), device=args.device, dtype=torch.bool)
#         self.node_mask = torch.zeros((self.tot_num, 29), device=args.device)
#         self.alpha = torch.zeros(self.tot_num, device=args.device)

#         tar_path = 'qm9/temp/qm9/dsgdb9nsd.xyz.tar.bz2'
#         tardata = tarfile.open(tar_path, 'r')
#         files = tardata.getmembers()
#         files = [file for idx, file in enumerate(files) if idx in file_idx_list]
#         i = 0
#         for file in files:
#             with tardata.extractfile(file) as openfile:
#                 # print(file.name)
#                 lines = [line.decode('utf-8') for line in openfile.readlines()]
#                 n = int(lines[0])
#                 smile = lines[n+3].split()[0]
#                 alpha = float(lines[1].split()[6])

#                 mol_xyz = lines[2:n+2]
#                 atom_positions = []
#                 atom_types = []
#                 for line in mol_xyz:
#                     atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
#                     atom_positions.append([float(posx), float(posy), float(posz)])
#                     atom_types.append(atom)
#                 # print(atom_positions)

#                 self.positions[i, 0:n] = torch.tensor(atom_positions)
#                 for j in range(n):
#                     self.node_mask[i, j] = 1
#                     self.one_hots[i, j, atom_encoder[atom_types[j]]] = 1
#                 self.alpha[i] = alpha 

#                 i += 1

#         # self.batch_size = 1
#         # self.tot_num = 1
#         self.batch_size = args.batch_size
#         print('finish dataloader init')

#     def __iter__(self):
#         return self

#     def __next__(self):

#         if self.i < self.tot_num:
#             positions = self.positions[self.i: self.i + self.batch_size]
#             one_hot = self.one_hots[self.i: self.i + self.batch_size]
#             node_mask = self.node_mask[self.i: self.i + self.batch_size]
#             alpha = self.alpha[self.i: self.i + self.batch_size]

#             bs, n_nodes = node_mask.size()
#             edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
#             diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
#             diag_mask = diag_mask.to(self.device)
#             edge_mask *= diag_mask
#             edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

#             data = {
#                 'positions': positions.detach(),
#                 'atom_mask': node_mask.detach(),
#                 'edge_mask': edge_mask.detach(),
#                 'one_hot': one_hot.detach(),
#                 'alpha': alpha.detach(),
#             }

#             self.i += self.batch_size
#             return data
#         else: 
#             self.i = 0
#             raise StopIteration

class QM9_srd_dataloader:
    def __init__(self, args):
        self.device = args.device
        # self.tot_num = 1
        # self.batch_size = 1
        with open('qm9_smiles_alpha.txt') as f:
            lines = f.readlines()
            smiles = [line.split()[0] for line in lines]
            alpha = [float(line.split()[1]) for line in lines]

        self.tot_num = len(smiles)
        self.positions = torch.zeros((self.tot_num, 29, 3), device=args.device)
        self.one_hots = torch.zeros((self.tot_num, 29, 5), device=args.device)
        self.node_mask = torch.zeros((self.tot_num, 29), device=args.device)
        self.alpha = torch.tensor(alpha, device=args.device)
        self.unvalid_flag = torch.zeros((self.tot_num), dtype=torch.bool, device=args.device)

        for i, smile in enumerate(smiles):
            # print(f'{i}_th data')
            pos, one_hot = smile_to_xyz(smile)
            if pos is None:
                self.unvalid_flag[i] = True
            else:
                node_num = pos.size(0)
                self.positions[i, :node_num] = pos
                self.one_hots[i, :node_num] = one_hot
                self.node_mask[i, :node_num] = 1

        self.i = 0
        self.batch_size = args.batch_size

        print('finish dataloader __init__')

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.tot_num:
            # positions = torch.tensor([[[-0.0233,  1.5153,  0.0190],
            #                         [-0.0060,  0.0068,  0.0140],
            #                         [ 0.5533, -0.9681, -1.0521],
            #                         [ 0.0166, -1.9454,  0.0072],
            #                         [ 0.6515, -0.9741,  1.0168],
            #                         [-1.1893, -0.9922,  0.0674],
            #                         [ 0.9933,  1.9206, -0.0281],
            #                         [-0.5803,  1.9052, -0.8399],
            #                         [-0.4966,  1.9004,  0.9289],
            #                         [ 0.0331, -1.0372, -2.0123],
            #                         [ 1.6382, -1.0214, -1.1847],
            #                         [ 1.7441, -1.0279,  1.0458],
            #                         [ 0.2244, -1.0489,  2.0215],
            #                         [-1.7595, -1.0685,  0.9983],
            #                         [-1.8450, -1.0629, -0.8058],
            #                         [0, 0, 0]]])
            # one_hot = torch.tensor([[[0., 1., 0., 0., 0.],
            #                         [0., 1., 0., 0., 0.],
            #                         [0., 1., 0., 0., 0.],
            #                         [0., 0., 1., 0., 0.],
            #                         [0., 1., 0., 0., 0.],
            #                         [0., 1., 0., 0., 0.],
            #                         [1., 0., 0., 0., 0.],
            #                         [1., 0., 0., 0., 0.],
            #                         [1., 0., 0., 0., 0.],
            #                         [1., 0., 0., 0., 0.],
            #                         [1., 0., 0., 0., 0.],
            #                         [1., 0., 0., 0., 0.],
            #                         [1., 0., 0., 0., 0.],
            #                         [1., 0., 0., 0., 0.],
            #                         [1., 0., 0., 0., 0.],
            #                         [0, 0, 0, 0, 0]]])
            # node_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
            # unvalid_flag = torch.zeros(len(one_hot), dtype=torch.bool)
            # alpha = torch.tensor([57.23])
            positions = self.positions[self.i: self.i + self.batch_size]
            one_hot = self.one_hots[self.i: self.i + self.batch_size]
            node_mask = self.node_mask[self.i: self.i + self.batch_size]
            unvalid_flag = self.unvalid_flag[self.i: self.i + self.batch_size]
            alpha = self.alpha[self.i: self.i + self.batch_size]

            bs, n_nodes = node_mask.size()
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
            diag_mask = diag_mask.to(self.device)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

            data = {
                'positions': positions.detach(),
                'atom_mask': node_mask.detach(),
                'edge_mask': edge_mask.detach(),
                'one_hot': one_hot.detach(),
                'unvalid_flag': unvalid_flag.detach(),
                'alpha': alpha.detach(),
            }

            self.i += self.batch_size
            return data
        else: 
            self.i = 0
            raise StopIteration


class DiffusionDataloader:
    # TODO add dim_mask
    def __init__(self, args_gen, model, nodes_dist, dims_dist, prop_dist, device, unkown_labels=False,
                 batch_size=1, iterations=200):
        self.args_gen = args_gen
        self.model = model
        self.nodes_dist = nodes_dist
        self.dims_dist = dims_dist
        self.prop_dist = prop_dist
        self.batch_size = batch_size
#         self.batch_size = 1
        self.iterations = iterations
        self.device = device
        self.unkown_labels = unkown_labels
        self.dataset_info = get_dataset_info(self.args_gen.dataset, self.args_gen.remove_h)
        self.i = 0

    def __iter__(self):
        return self

    def sample(self):
        # TODO: dims_mask
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        dimsxsample = self.dims_dist.sample(self.batch_size)
#         dimsxsample = nodesxsample.clone()

        # print(nodesxsample, dimsxsample)

        context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        one_hot, charges, x, node_mask = sample(self.args_gen, self.device, self.model,
                                                self.dataset_info, self.prop_dist, nodesxsample=nodesxsample,
                                                context=context)
        
        dims_mask = torch.zeros((len(nodesxsample), self.dataset_info['max_n_dims']), device=x.device)
        for i in range(len(dimsxsample)):
            dims_mask[i, 0:dimsxsample[i]] = 1
        dims_mask = dims_mask.unsqueeze(1)
        x = x * dims_mask

        atom_types = one_hot.argmax(dim=2)
        # TODO convert srd positions to real positions
        smiles = srd_to_smiles(x, node_mask, atom_types)

        positions = torch.zeros((len(nodesxsample), self.dataset_info['max_n_nodes'], 3), device=x.device)
        unvalid_flag = torch.zeros(len(nodesxsample), dtype=torch.bool, device=x.device)

        for i in range(self.batch_size):
#             print(smiles[i])
            # todo: test
            if smiles[i] is None or '.' in smiles[i]:
#             if smiles[i] is None:
                unvalid_flag[i] = True
            else:
                position, tmp_one_hot = smile_to_xyz(smiles[i])
                if position is None:
                    unvalid_flag[i] = True
                else:
                    assert(position.size(0) <= self.dataset_info['max_n_nodes'])
                    positions[i, 0:position.size(0)] = position
                    one_hot[i, 0:position.size(0)] = tmp_one_hot

        check_mask(x, node_mask, dims_mask)
        node_mask = node_mask.squeeze(2)

        context = context.squeeze(1)

        # edge_mask
        bs, n_nodes = node_mask.size()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.to(self.device)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

        prop_key = self.prop_dist.properties[0]
        if self.unkown_labels:
            context[:] = self.prop_dist.normalizer[prop_key]['mean']
        else:
            context = context * self.prop_dist.normalizer[prop_key]['mad'] + self.prop_dist.normalizer[prop_key]['mean']
        data = {
            'positions': positions.detach(),
            'atom_mask': node_mask.detach(),
            'edge_mask': edge_mask.detach(),
            'one_hot': one_hot.detach(),
            'unvalid_flag': unvalid_flag.detach(),
            # TODO: tmp
            'dim_mask': dims_mask.squeeze(1).detach(),
            'smiles': smiles, 
            prop_key: context.detach()
        }
        return data

    def __next__(self):
        if self.i <= self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations


def main_quantitative(args):
    # Get classifier
    #if args.task == "numnodes":
    #    class_dir = args.classifiers_path[:-6] + "numnodes_%s" % args.property
    #else:
    class_dir = args.classifiers_path
    classifier = get_classifier(class_dir).to(args.device)

    # Get generator and dataloader used to train the generator and evalute the classifier
    args_gen = get_args_gen(args.generators_path)

    # Careful with this -->
    if not hasattr(args_gen, 'diffusion_noise_precision'):
        args_gen.normalization_factor = 1e-4
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    dataloaders = get_dataloader(args_gen)
    print(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path, dataloaders,
                                                    args.device, args_gen, property_norms)
    histogram = dataset_info['ranks'] 
    dims_dist = DistributionNodes(histogram)

    # Create a dataloader with the generator

    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    if args.task == 'edm':
        diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, dims_dist, prop_dist,
                                                   args.device, batch_size=args.batch_size, iterations=args.iterations)
        print("EDM: We evaluate the classifier on our generated samples")
        loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
        print("Loss classifier on Generated samples: %.4f" % loss)
    elif args.task == 'qm9_second_half':
        print("qm9_second_half: We evaluate the classifier on QM9")
        loss = test(classifier, 0, dataloaders['test'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on qm9_second_half: %.4f" % loss)
    elif args.task == 'srd_qm9':
        print("srd_qm9")
        dataloader = QM9_srd_dataloader(args)
        loss = test(classifier, 0, dataloader, mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on srd_qm9: %.4f" % loss)
    elif args.task == 'my_qm9':
        print("my qm9")
        dataloader = QM9_dataloader(args)
        loss = test(classifier, 0, dataloader, mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on my_qm9: %.4f" % loss)



#     elif args.task == 'naive':
#         print("Naive: We evaluate the classifier on QM9")
#         dataset_type = 'train'
#         length = dataloaders[dataset_type].dataset.data[args.property].size(0)
#         idxs = torch.randperm(length)
#         dataloaders[dataset_type].dataset.data[args.property] = dataloaders[dataset_type].dataset.data[args.property][idxs]
#         loss = test(classifier, 0, dataloaders[dataset_type], mean, mad, args.property, args.device, args.log_interval,
#                     args.debug_break)
#         print("Loss classifier on naive: %.4f" % loss)
    #elif args.task == 'numnodes':
    #    print("Numnodes: We evaluate the numnodes classifier on EDM samples")
    #    diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist, device,
    #                                               batch_size=args.batch_size, iterations=args.iterations)
    #    loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
    #    print("Loss numnodes classifier on EDM generated samples: %.4f" % loss)


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/analysis/run%s/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    vis.visualize_chain("outputs/%s/analysis/run%s/" % (args.exp_name, epoch), dataset_info,
                        wandb=None, mode='conditional', spheres_3d=True)

    return one_hot, charges, x


def main_qualitative(args):
    args_gen = get_args_gen(args.generators_path)
    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path,
                                                               dataloaders, args.device, args_gen,
                                                               property_norms)

    for i in range(args.n_sweeps):
        print("Sampling sweep %d/%d" % (i+1, args.n_sweeps))
        save_and_sample_conditional(args_gen, device, model, prop_dist, dataset_info, epoch=i, id_from=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_alpha')
    parser.add_argument('--generators_path', type=str, default='outputs/exp_cond_alpha_pretrained')
    parser.add_argument('--classifiers_path', type=str, default='qm9/property_prediction/outputs/exp_class_alpha_pretrained')
    parser.add_argument('--property', type=str, default='alpha',
                        help="'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--debug_break', type=eval, default=False,
                        help='break point or not')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='break point or not')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='break point or not')
    parser.add_argument('--iterations', type=int, default=20,
                        help='break point or not')
    parser.add_argument('--task', type=str, default='qualitative',
                        help='naive, edm, qm9_second_half, qualitative')
    parser.add_argument('--n_sweeps', type=int, default=10,
                        help='number of sweeps for the qualitative conditional experiment')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    if args.task == 'qualitative':
        main_qualitative(args)
    else:
        main_quantitative(args)