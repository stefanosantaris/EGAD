from algo.EGAD import EGAD
from evaluate.MLPEvaluation import MLPEvaluation
from utils.dataset.GraphLoader import GraphLoader
import torch
import torch.optim as optim
from torch import nn
import scipy.sparse as sps
import numpy as np
import random

from utils.utils import sparse_mx_to_torch_sparse_tensor, sparse_to_tuple


class Trainer():
    def __init__(self, exp_params, cuda_enabled):
        super(Trainer, self).__init__()
        self.graph_loader = GraphLoader(exp_params['path'] + "/" + exp_params['extract_folder'] + "/")
        if cuda_enabled == 0:
            self.device = "cpu"
        else:
            self.device = 'cuda'


    def normalize(self, adj):
        adj_with_diag = adj + sps.identity(adj.shape[0], dtype=np.float32).tocsr()
        rowsum = np.array(adj_with_diag.sum(1))
        degree_mat_inv_sqrt = sps.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_with_diag.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo().astype(np.float32)
        return adj_normalized

    def prepare_test_adj(self, input_graph, ground_truth_adj):
        coords, values, shape = sparse_to_tuple(input_graph)
        ground_truth_adj = (ground_truth_adj[:input_graph.shape[0], :input_graph.shape[1]]).todense()
        for coord in coords:
            ground_truth_adj[coord[0], coord[1]] = 0.
            ground_truth_adj[coord[1], coord[0]] = 0.
        return sps.csr_matrix(ground_truth_adj, dtype=float)

    def construct_dataset(self, graph, window_size, negative_sample):
        start_graph = max(0, graph - window_size)
        max_id = 0
        for i in range(start_graph, graph + 1):
            adj = self.graph_loader.read_adjacency(i, max_id)
            max_id = adj.shape[0] - 1

        train_adj = []
        total_train_edges = np.zeros((max_id + 1, max_id + 1))
        for i in range(start_graph, graph + 1):
            adj = self.graph_loader.read_adjacency(i, max_id)
            train_adj_dense = adj.todense()
            train_adj_dense = np.where(train_adj_dense > 0.2, train_adj_dense, 0)
            train_adj_sparse = sps.csr_matrix(train_adj_dense)
            coords, values, shape = sparse_to_tuple(train_adj_sparse)
            for coord in coords:
                total_train_edges[coord[0], coord[1]] = 1
            train_adj.append(train_adj_sparse)

        train_adj_norm = []
        features = []
        for i, adj in enumerate(train_adj):
            train_adj_norm.append(sparse_mx_to_torch_sparse_tensor(self.normalize(adj)).to(device=self.device))
            features.append(torch.tensor(sps.identity(adj.shape[0], dtype=np.float32, format='coo').todense(),
                                         dtype=torch.float32).to_sparse().to(device=self.device))

            # Generate the train_adj_label with negative sampling
            if i == len(train_adj) - 1:
                # Construct a full matrix with ones to generate negative sample tuples
                train_ns = np.ones_like(adj.todense()) - total_train_edges - sps.identity(total_train_edges.shape[0])

                ns_coord, ns_values, ns_shape = sparse_to_tuple(train_ns)

                train_coord, train_values, train_shape = sparse_to_tuple(adj)
                train_label_ind = np.zeros_like(adj.todense())

                for coord in train_coord:
                    train_label_ind[coord[0], coord[1]] = 1

                sequence = [i for i in range(len(ns_coord))]
                random_coords = set(random.sample(sequence, negative_sample * len(train_coord)))

                for coord in random_coords:
                    train_label_ind[ns_coord[coord][0], ns_coord[coord][1]] = 1

                train_adj_dense = adj.todense()
                nnz_ind = np.nonzero(train_label_ind)
                train_label_val = train_adj_dense[nnz_ind][0]

                train_adj_label = torch.reshape(
                    torch.tensor(train_label_val, dtype=torch.float32).to(device=self.device), (-1,))
                train_adj_ind = [torch.tensor(nnz_ind[0], requires_grad=False).to(device=self.device),
                                 torch.tensor(nnz_ind[1], requires_grad=False).to(device=self.device)]

        test_adj_dense = self.prepare_test_adj(total_train_edges,
                                               self.graph_loader.read_adjacency(graph + 1, max_id)).todense()
        test_adj_high = np.where(test_adj_dense > 0.2, test_adj_dense, 0)
        test_adj_ind = np.where(test_adj_high > 0., 1, 0)
        nnz_ind = np.nonzero(test_adj_ind)

        test_adj = torch.tensor(test_adj_high[nnz_ind], dtype=torch.float32, requires_grad=False).to(device=self.device)
        test_adj_ind = [torch.tensor(nnz_ind[0], requires_grad=False).to(device=self.device),
                        torch.tensor(nnz_ind[1], requires_grad=False).to(device=self.device)]

        return train_adj_norm, train_adj_label, train_adj_ind, features, test_adj, test_adj_ind

    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_model(self, emb_size, train_embeddings, train_values, test_embeddings, test_values):
        evaluateModel = MLPEvaluation(emb_size).to(device=self.device)
        evaluateOptimizer = optim.Adam(evaluateModel.parameters(), lr=0.001)
        for epoch in range(10):
            evaluateModel.train()
            evaluateOptimizer.zero_grad()
            output = evaluateModel(train_embeddings)
            criterion = nn.MSELoss()
            loss = criterion(output.squeeze(), train_values)
            loss.backward()
            evaluateOptimizer.step()

        evaluateModel.eval()
        final_output = evaluateModel(test_embeddings)
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        mae_score = mae_criterion(final_output.squeeze(), test_values).detach().to(device='cpu').numpy()
        rmse_score = torch.sqrt(criterion(final_output.squeeze(), test_values)).detach().to(device='cpu').numpy()
        return mae_score, rmse_score

    def get_edge_embeddings(self, embeddings, indices):
        src_embeddings = torch.index_select(embeddings, 0, indices[0])
        dst_embeddings = torch.index_select(embeddings, 0, indices[1])
        return torch.mul(src_embeddings, dst_embeddings)

    def train_model(self, args):
        num_exp = args.num_exp
        start_graph = args.start_graph
        end_graph = args.end_graph
        window_size = args.window
        dropout = args.dropout
        alpha = args.alpha
        learning_rate = args.learning_rate
        negative_sample = args.ns

        teacher_n_heads = args.teacher_n_heads
        teacher_embed_dim = args.teacher_embed_size

        student_embed_dim = args.student_emb
        student_n_heads = args.student_heads



        results = {}
        print("Start training")
        for graph in range(start_graph, end_graph + 1):
            results[graph] = {'teacher':
                                  {'num_params':0, 'mae':0., 'rmse':0.},
                              'student':
                                  {'num_params': 0, 'mae': 0., 'rmse': 0.}
                              }
            teacher_mae = []
            teacher_rmse = []
            teacher_number_of_params = []

            student_mae = []
            student_rmse = []
            student_number_of_params = []
            train_adj_norm, train_adj_label, train_adj_ind, features, test_adj, test_adj_ind = self.construct_dataset(graph, window_size, negative_sample)
            for i in range(num_exp):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Experiment ", i)
                num_cells = len(train_adj_norm)
                teacher_model = EGAD(num_cells, features[0].shape[0], 2 * teacher_embed_dim, teacher_embed_dim, teacher_n_heads, dropout, alpha).to(device=self.device)
                model_params = self.count_parameters(teacher_model)
                teacher_number_of_params.append(model_params)
                optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)
                for epoch in range(100):
                    teacher_model.train()
                    optimizer.zero_grad()
                    output = teacher_model(features, train_adj_norm)
                    reconstruction = torch.sigmoid(torch.mm(output, torch.t(output)))

                    reconstructed_val = reconstruction[train_adj_ind]

                    predicted = reconstructed_val
                    target = train_adj_label

                    criterion = nn.MSELoss()

                    R_loss = criterion(predicted, target)
                    loss_train = torch.sqrt(R_loss)
                    loss_train.backward()

                    optimizer.step()
                print("Teacher finished")


                teacher_model.eval()

                final_output = teacher_model(features, train_adj_norm)
                train_embeddings = self.get_edge_embeddings(final_output, train_adj_ind).detach().to(device=self.device)
                test_embeddings = self.get_edge_embeddings(final_output, test_adj_ind).detach().to(device=self.device)


                mae_score, rmse_score = self.evaluate_model(teacher_embed_dim, train_embeddings, train_adj_label, test_embeddings,
                                                            test_adj)
                teacher_mae.append(mae_score)
                teacher_rmse.append(rmse_score)

                print("TEACHER FINISHED for GRAPH {} and EXP {}".format(graph, i))

                ##### STUDENT
                if args.distillation == 1:

                    student_model = EGAD(num_cells, features[0].shape[0], 2 * student_embed_dim, student_embed_dim,
                                                  student_n_heads, dropout, alpha).to(device=self.device)
                    model_params = self.count_parameters(student_model)
                    student_number_of_params.append(model_params)
                    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
                    for epoch in range(100):
                        student_model.train()
                        optimizer.zero_grad()
                        output = student_model(features, train_adj_norm)
                        reconstruction = torch.sigmoid(torch.mm(output, torch.t(output)))

                        teacher_output = teacher_model(features, train_adj_norm)
                        teacher_reconstruction = torch.sigmoid(torch.mm(teacher_output, torch.t(teacher_output)))

                        student_reconstructed_val = reconstruction[train_adj_ind]
                        teacher_reconstruction_val = teacher_reconstruction[train_adj_ind]

                        criterion = nn.MSELoss()
                        student_R_loss = criterion(student_reconstructed_val, train_adj_label) + criterion(teacher_reconstruction_val, train_adj_label)
                        loss_train = torch.sqrt(student_R_loss)
                        loss_train.backward()

                        optimizer.step()

                    student_model.eval()
                    final_output = student_model(features, train_adj_norm)
                    train_embeddings = self.get_edge_embeddings(final_output, train_adj_ind).detach().to(
                        device=self.device)
                    test_embeddings = self.get_edge_embeddings(final_output, test_adj_ind).detach().to(
                        device=self.device)

                    mae_score, rmse_score = self.evaluate_model(student_embed_dim, train_embeddings, train_adj_label,
                                                                test_embeddings,
                                                                test_adj)


                    student_mae.append(mae_score)
                    student_rmse.append(rmse_score)





            results[graph]['teacher']['num_params'] = np.mean(teacher_number_of_params)
            results[graph]['teacher']['mae'] = np.mean(teacher_mae)
            results[graph]['teacher']['rmse'] = np.mean(teacher_rmse)
            if args.distillation == 1:
                results[graph]['student']['num_params'] = np.mean(student_number_of_params)
                results[graph]['student']['mae'] = np.mean(student_mae)
                results[graph]['student']['rmse'] = np.mean(student_rmse)
            print("Graph {} : TEACHER N_PARAMS {} : TEACHER MAE {} : TEACHER RMSE {} : STUDENT N_PARAMS {} : STUDENT MAE {} : STUDENT RMSE {}".format(graph,
                                                                                                                                                      results[graph]['teacher']['num_params'],
                                                                                                                                                      results[graph]['teacher']['mae'],
                                                                                                                                                      results[graph]['teacher']['rmse'],
                                                                                                                                                      results[graph]['student']['num_params'],
                                                                                                                                                      results[graph]['student']['mae'],
                                                                                                                                                      results[graph]['student']['rmse']
                                                                                                                                                      ))
        return results


