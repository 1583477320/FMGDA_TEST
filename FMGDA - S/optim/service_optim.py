import torch.optim as optim
import torch
from methods.WeightedMethod import MultiObjectiveWeightOptimizer
from utils.options import overwrite_grad

'''
服务端共享层更新梯度
'''


class ServicAgg():
    def __init__(self, args, ServiceModel, client_gard):
        self.device = torch.device(
            'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        self.ServiceModel = ServiceModel.to(self.device)
        self.args = args
        self.client_gard = client_gard

    def get_last_model_parm(self):
        '''
        获取该轮模型参数，用于下次计算梯度时使用其上次模型的梯度
        '''
        last_model_parm = self.ServiceModel.shared_parameters.state_dict()
        return last_model_parm

    def get_last_model_gard(self):
        '''
        获取该轮每一个任务的dt，用于计算下一次梯度时使用上轮的任务梯度
        '''
        return sum(self.client_gard) / len(self.client_gard)

    def weighted(self, client_gard):
        '''
        求权重函数
        '''
        weighted_optimizer = MultiObjectiveWeightOptimizer()
        w = weighted_optimizer.optimize(client_gard)
        return w

    def federated_aggregation(self, client_gard, w):
        gt = torch.sum(client_gard * w, dim=1)
        return gt

    def backward(self):
        ServiceModel = self.ServiceModel
        shared_parameters = list((p for n, p in ServiceModel.named_parameters() if "task" not in n))
        optimizer = optim.SGD(ServiceModel.parameters(), lr=self.args.global_lr, momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)

        # 求权重
        w = self.weighted(self.client_gard)

        # 返回权重求和后的向量梯度
        gt = torch.mv(sum(self.client_gard) / len(self.client_gard), w)

        # 返回各任务梯度,用于计算下一次梯度时使用的上轮的任务梯度
        dt = self.get_last_model_gard()

        # 更新服务端参数
        optimizer.zero_grad()
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        overwrite_grad(shared_parameters, gt, grad_dims)
        optimizer.step()

        return dt, ServiceModel
