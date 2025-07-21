import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from utils.data_load import generate_multi_mnist, split_data_to_servers,train_dataset,test_dataset
from model.ClientModel import ClientMTLModel
from model.ServiceModel import ServerSharedModel
from optim.client_optim import ClientAgg
from optim.service_optim import ServicAgg
from utils.options import args_parser, last_client_init, check_models_parameters_identical
import pandas as pd
import copy
import numpy as np
import random
from tqdm import tqdm
import time


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    # args.batch_size = 256
    torch.manual_seed(args.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.

    # 准备原始数据集
    # 不同分类生成一个批次
    # train_dataset = generate_multi_mnist(num_samples=6000)
    train_dataset = train_dataset

    # 生成测试数据
    # test_dataset = generate_multi_mnist(num_samples=1000, train=False)
    test_dataset = test_dataset

    # dataset_size = len(train_dataset)
    # train_size = int(dataset_size * 0.8)  # 80% 训练集
    # val_size = dataset_size - train_size  # 20% 验证集
    # # 划分数据集
    # train_subset, test_dataset = random_split(
    #     train_dataset,
    #     [train_size, val_size],
    #     generator=torch.Generator().manual_seed(42)  # 保证可重复性
    # )

    client_datasets = split_data_to_servers(train_dataset, num_servers=args.num_clients)  # 将训练集分给客户端

    # loss_history = {'task1': {'fmgda': [], 'fmgda_s': []}, 'task2': {'fmgda': [], 'fmgda_s': []}}
    # loss_history = {'task1': {'16':[], '64':[], '128': [], '256': []}, 'task2': {'16':[], '64':[], '128': [], '256': []}}
    loss_history = {'task1': { '256': []}
                    ,'task2': {'256': []}
                    }
    # loss_eval = {'task1': {'fmgda': [], 'fmgda_s': []}, 'task2': {'fmgda': [], 'fmgda_s': []}}
    # loss_eval = {'task1': {'16':[], '64':[], '128': [], '256': []}, 'task2': {'16':[], '64':[], '128': [], '256': []}}
    loss_eval = {'task1': {'256': []}
                 ,'task2': {'256': []}
                 }
    # acc = {'task1': {'fmgda': [], 'fmgda_s': []}, 'task2': {'fmgda': [], 'fmgda_s': []}}
    # acc = {'task1': {'16':[], '64':[], '128': [], '256': []}, 'task2': {'16':[], '64':[], '128': [], '256': []}}
    acc = {'task1': {'256': []}
           ,'task2': {'256': []}
           }

    for batch_size in [256]:
        # 初始化模型参数
        args.batch_size = batch_size
        server_model = ServerSharedModel()
        client_model = ClientMTLModel(server_model)
        print(client_model)
        model_total_params = [p.numel() for p in client_model.parameters() if p.requires_grad]
        print('model trainable parameter count: sum{}={}\n'.format(model_total_params, sum(model_total_params)))
        clients_model = [copy.deepcopy(client_model) for _ in range(args.num_clients)]  # 模型初始化


        # copy weights
        old_global_weights = client_model.state_dict()

        # method == fmgda_s时对下列参数初始化
        last_clients_model = [copy.deepcopy(client_model) for _ in range(args.num_clients)]
        last_client_grads = last_client_init(client_model)

        for epoch in tqdm(range(1, args.global_epochs + 1)):
            client_models_gard, task1_loss_locals, task2_loss_locals=[],[],[]
            print(f'\n | Global Training Round : {epoch} |\n')

            for client_idx, dataset in client_datasets.items():
                # 加载本地数据
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

                client_model = clients_model[client_idx]
                client_model.train()
                if args.method == 'fmgda_s':
                    client_local = ClientAgg(args.method, args, client_model, train_loader,
                                             # last_client_model=last_client_model,
                                             last_client_grads=last_client_grads)
                else:
                    client_local = ClientAgg(args.method, args, client_model, train_loader, global_round=epoch)
                client_model, w, task_loss = client_local.backward()

                # 客户端梯度记录
                clients_model[client_idx] = client_model
                client_models_gard.append(copy.deepcopy(w))

                # 客户端loss记录
                task1_loss_locals.append(copy.deepcopy(task_loss[0].item()))
                task2_loss_locals.append(copy.deepcopy(task_loss[1].item()))

            # 服务端共享层参数更新
            servicagg = ServicAgg(args, server_model, client_models_gard)
            if args.method == 'fmgda_s':
                last_shared_parameters = servicagg.get_last_model_parm()
            else:
                pass
            last_client_grads, server_model = servicagg.backward()  # last_client_shared_parameters返回的结构为向量结构

            task1_loss_avg = sum(task1_loss_locals) / len(task1_loss_locals)
            task2_loss_avg = sum(task2_loss_locals) / len(task2_loss_locals)
            loss_history['task1']['{}'.format(batch_size)].append(task1_loss_avg)
            loss_history['task2']['{}'.format(batch_size)].append(task2_loss_avg)

            # 更新客户端共享层模型
            new_global_weights = copy.deepcopy(server_model.shared_parameters.state_dict())
            for i in range(args.num_clients):
                clients_model[i].shared_layer.load_state_dict(new_global_weights)

            # 检查参数（有50%概率检测到相同模型）
            try:
                check_models_parameters_identical(clients_model)
                print("随机选择的两个模型参数不同")
            except RuntimeError as e:
                print(f"检测到相同参数: {str(e)}")

            # 测试
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            criterion = nn.CrossEntropyLoss().to(args.device)

            task1_acc_list, task2_acc_list, task1_loss_eval, task2_loss_eval = [], [], [], []
            for client_idx in range(args.num_clients):
                client_model = clients_model[client_idx]
                client_model.eval()
                loss1,loss2, total, correct1, correct2 = 0.0, 0.0, 0.0, 0.0,0.0

                with torch.no_grad():
                    for batch_idx, (images, labels) in enumerate(test_loader):
                        images, labels0,labels1  = images.to(args.device), labels[0].to(args.device), labels[1].to(args.device)
    
                        # Inference
                        outputs = client_model(images)
                        batch_loss1 = criterion(outputs[0], labels0)
                        batch_loss2 = criterion(outputs[1], labels1)
                        
                        loss1 += batch_loss1.item()
                        loss2 += batch_loss2.item()
    
                        # Prediction
                        _, preds1 = torch.max(outputs[0], 1)
                        _, preds2 = torch.max(outputs[1], 1)
            
                        correct1 += preds1.eq(labels0).sum().item()
                        correct2 += preds2.eq(labels1).sum().item()
                        total += labels1.size(0)

                avg_loss1 = loss1 / len(test_loader)
                avg_loss2 = loss2 / len(test_loader)
                task1_accuracy = 100. * correct1 / total
                task2_accuracy = 100. * correct2 / total
                
                task1_loss_eval.append(avg_loss1)
                task2_loss_eval.append(avg_loss2)

                task1_acc_list.append(task1_accuracy)
                task2_acc_list.append(task2_accuracy)

            task1_loss_eval_avg = sum(task1_loss_eval)/len(task1_loss_eval)
            loss_eval['task1']['{}'.format(batch_size)].append(task1_loss_eval_avg)

            task2_loss_eval_avg = sum(task2_loss_eval) / len(task2_loss_eval)
            loss_eval['task2']['{}'.format(batch_size)].append(task2_loss_eval_avg)

            task1_acc_avg = sum(task1_acc_list)/len(task1_acc_list)
            acc['task1']['{}'.format(batch_size)].append(task1_acc_avg)

            task2_acc_avg = sum(task2_acc_list) / len(task2_acc_list)
            acc['task2']['{}'.format(batch_size)].append(task2_acc_avg)
    # 保存模型参数/kaggle/working
    torch.save(clients_model[0].state_dict(), './client0_model_model_params.pth')
    torch.save(server_model.state_dict(), './server_model_params.pth')

    # 数据保存
    # 将字典转换为扁平结构
    data1 = {}
    for task, methods in loss_history.items():
        for method, values in methods.items():
            col_name = f"{task}_{method}"  # 组合列名：任务_方法
            data1[col_name] = values

    df1 = pd.DataFrame(data1)
    df1.to_excel("./loss_history.xlsx", index=False)
    #
    # # 将字典转换为扁平结构
    data2 = {}
    for task, methods in loss_eval.items():
        for method, values in methods.items():
            col_name = f"{task}_{method}"  # 组合列名：任务_方法
            data2[col_name] = values

    df2 = pd.DataFrame(data2)
    df2.to_excel("./loss_eval.xlsx", index=False)
    #
    # # 将字典转换为扁平结构
    data3 = {}
    for task, methods in acc.items():
        for method, values in methods.items():
            col_name = f"{task}_{method}"  # 组合列名：任务_方法
            data3[col_name] = values

    df3 = pd.DataFrame(data3)
    df3.to_excel("./acc.xlsx", index=False)

    # 绘制训练集损失曲线
    fig, ax = plt.subplots()
    task1_loss = loss_history["task1"]
    for i in [256]:
        ax.semilogy(range(len(task1_loss["{}".format(i)])), task1_loss["{}".format(i)], label="{}".format(i))
    ax.spines['left'].set_position('zero')  # Y轴在x=0处
    ax.set_xlim(left=0)  # 只显示x正半轴
    ax.set_title("Task1 Train Loss")
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("Train Local Loss")
    # ax.set_yticks([0.01, 0.1, max(max(task1_loss["128"]), max(task1_loss["256"]))])
    ax.set_yticks([0.01, 0.1, max(task1_loss["256"])])
    plt.legend()
    plt.grid(False)
    # 保存图像
    plt.savefig('./task1_mulit_loss_curve.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots()
    task2_loss = loss_history["task2"]
    for i in [256]:
        ax.semilogy(range(len(task2_loss["{}".format(i)])), task2_loss["{}".format(i)], label="{}".format(i))
    ax.spines['left'].set_position('zero')  # Y轴在x=0处
    ax.set_xlim(left=0)  # 只显示x正半轴
    ax.set_title("Task2 Train Loss")
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("Train Local Loss")
    # ax.set_yticks([0.01, 0.1, max(max(task2_loss["128"]), max(task2_loss["256"]))])
    ax.set_yticks([0.01, 0.1, max(task1_loss["256"])])
    plt.legend()
    plt.grid(False)
    # 保存图像
    plt.savefig('./task2_mulit_loss_curve.png', dpi=300, bbox_inches='tight')

    # 绘制test集损失曲线
    fig, ax = plt.subplots()
    test_task1_loss = loss_eval["task1"]
    for i in [256]:
        ax.semilogy(range(len(test_task1_loss["{}".format(i)])), test_task1_loss["{}".format(i)], label="{}".format(i))
    ax.spines['left'].set_position('zero')  # Y轴在x=0处
    ax.set_xlim(left=0)  # 只显示x正半轴
    ax.set_title("Task1 Test Loss")
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("Test Local Loss")
    # ax.set_yticks([0.01, 0.1, max(max(test_task1_loss["128"]), max(test_task1_loss["256"]))])
    ax.set_yticks([0.01, 0.1, max(task1_loss["256"])])
    plt.legend()
    plt.grid(False)
    # 保存图像
    plt.savefig('./test_task1_mulit_loss_curve.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots()
    test_task2_loss = loss_eval["task2"]
    for i in [256]:
        ax.semilogy(range(len(test_task2_loss["{}".format(i)])), test_task2_loss["{}".format(i)], label="{}".format(i))
    ax.spines['left'].set_position('zero')  # Y轴在x=0处
    ax.set_xlim(left=0)  # 只显示x正半轴
    ax.set_title("Task2 Test Loss")
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("Test Local Loss")
    # ax.set_yticks([0.01, 0.1, max(max(test_task2_loss["128"]), max(test_task2_loss["256"]))])
    ax.set_yticks([0.01, 0.1, max(task1_loss["256"])])
    plt.legend()
    plt.grid(False)
    # 保存图像
    plt.savefig('./test_task2_mulit_loss_curve.png', dpi=300, bbox_inches='tight')

    # 绘制test集acc曲线
    fig, ax = plt.subplots()
    task1_acc = acc["task1"]
    for i in [256]:
        ax.plot(range(len(task1_acc["{}".format(i)])), task1_acc["{}".format(i)], label="{}".format(i))
    ax.spines['left'].set_position('zero')  # Y轴在x=0处
    ax.set_xlim(left=0)  # 只显示x正半轴
    ax.set_title("Task1 Test Acc")
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("Test Acc")
    # ax.set_yticks([1, 10, max(max(task1_acc["fmgda"]), max(task1_acc["fmgda_s"]))])
    # ax.set_yticks([1, 10, max(task1_acc["256"])])
    plt.legend()
    plt.grid(False)
    # 保存图像
    plt.savefig('./task1_acc_curve.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots()
    task2_acc = acc["task2"]
    for i in [256]:
        ax.plot(range(len(task2_acc["{}".format(i)])), task2_acc["{}".format(i)], label="{}".format(i))
    ax.spines['left'].set_position('zero')  # Y轴在x=0处
    ax.set_xlim(left=0)  # 只显示x正半轴
    ax.set_title("Task2 Test Acc")
    ax.set_xlabel("Communication rounds")
    ax.set_ylabel("Test Acc")
    # ax.set_yticks([1, 10, max(task2_acc["256"])])
    # ax.set_yticks([1, 10, max(task2_acc["fmgda"])])
    plt.legend()
    plt.grid(False)
    # 保存图像
    plt.savefig('./task2_acc_curve.png', dpi=300, bbox_inches='tight')

    plt.show()
