# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import CustomKNN


class clientLocalHead(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx


        distance = distances.CosineSimilarity()#LpDistance(normalize_embeddings=True, p=2, power=1)
        reducer = reducers.ThresholdReducer(low=0)
        self.loss_func = losses.TripletMarginLoss(margin=0.2,  distance=distance, reducer=reducer)
        self.mining_func = miners.TripletMarginMiner(
            margin=0.2, distance=distance, type_of_triplets="all" #"hard" is hard to learn, why?
        )

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                embeddings = self.model.hx(x)
                indices_tuple = self.mining_func(embeddings, y)
                loss1 = self.loss_func(embeddings, y, indices_tuple)

                output = self.model.clf(embeddings)
                loss2 = self.loss(output, y)

                self.optimizer.zero_grad()
                loss = loss1+loss2
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        

    def local_initialization(self, global_model):
        self.global_model = global_model
        params_g = list(global_model.parameters())
        params = list(self.model.parameters())
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()