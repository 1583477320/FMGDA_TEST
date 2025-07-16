import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from utils.data_load import generate_multi_mnist, split_data_to_servers
from model.ClientModel import ClientMTLModel
from model.ServiceModel import ServerSharedModel
from optim.client_optim import ClientAgg
from optim.service_optim import ServicAgg
from utils.options import args_parser, last_client_init
import pandas as pd
import copy
import numpy as np
import random

def train(args, server_model, clients_model, client_datasets, method: str,
          last_clients_model=None,
          last_client_grads=None, ):
    print(f"======== batch_size {args.batch_size} ========")
    print(f"==={args.method} Federal Round {epoch}/{args.global_epochs} ===")

    # 超参设置
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.method = method

    task1_loss_locals = []
    # task2_loss_locals = []

    # 客户端本地训练
    client_models_gard = []
    for client_idx, dataset in client_datasets.items():
        client_model = clients_model[client_idx]
        if args.method == 'fmgda_s':
            last_client_model = last_clients_model[client_idx]
            last_clients_model[client_idx] = client_model
        else:
            pass
        # 加载本地数据
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        # 本地多任务训练
        # client_model = ClientMTLModel(server_model).to(args.device)
        if args.method == 'fmgda_s':
            client_local = ClientAgg(args.method, args, client_model, train_loader,
                                     last_client_model=last_client_model,
                                     last_client_grads=last_client_grads)
        else:
            client_local = ClientAgg(args.method, args, client_model, train_loader)
        client_model, client_gard, task_loss = client_local.backward()

        clients_model[client_idx] = client_model
        client_models_gard.append(client_gard)

        # 记录客户端各任务loss
        if client_idx == 0:
            task1_loss_locals.append(task_loss[0].item())
            # task2_loss_locals.append(task_loss[1].item())

    train1_loss_avg = sum(task1_loss_locals) / len(task1_loss_locals)
    # train2_loss_avg = sum(task2_loss_locals) / len(task2_loss_locals)

    # 服务端共享层参数更新
    servicagg = ServicAgg(args, server_model, client_models_gard)
    if args.method == 'fmgda_s':
        last_shared_parameters = servicagg.get_last_model_parm()
    else:
        pass
    last_client_grads, server_model = servicagg.backward()  # last_client_shared_parameters返回的结构为向量结构

    # 更新客户端共享层模型
    for i in range(args.num_clients):
        clients_model[i].shared_layer.load_state_dict(server_model.shared_parameters.state_dict())

    print(
        "train1 loss:{:.4f}".format(train1_loss_avg),
        # "train2 loss:{:.4f}".format(train2_loss_avg)
    )
    print("----------------------------------------------")

    if args.method == 'fmgda_s':
        return server_model, clients_model, last_clients_model, last_client_grads, (train1_loss_avg
                                                                                    # , train2_loss_avg
                                                                                    )
    else:
        return server_model, clients_model, (train1_loss_avg
                                             # ,train2_loss_avg
                                             )


def test(args, clients_model, test_data):
    # 评估全局模型（以客户端0为例）
    criterion = nn.CrossEntropyLoss()
    client0_model = clients_model[0].to(args.device)
    client0_model.eval()

    # 记录acc
    total_correct_task1 = 0
    total_correct_task2 = 0

    # 记录损失
    test_losses = {i: 0.0 for i in range(args.num_tasks)}
    train_loader_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    with torch.no_grad():
        for data, targets in train_loader_test:
            # data, targets[0], targets[1] = data.to(args.device), targets[0].to(args.device), targets[1].to(
            #     args.device)
            data, targets[0] = data.to(args.device), targets[0].to(args.device)
            outputs = client0_model(data)

            # loss
            # total_loss_task1 += criterion(pred_task1, target_task1).item()
            # total_loss_task2 += criterion(pred_task2, target_task2).item()
            # losses = torch.stack([
            #     criterion(outputs[i], targets[i])
            #     for i in range(args.num_tasks)
            # ])
            losses = torch.stack([criterion(outputs, targets[0])])
            # correct
            pred1 = outputs.argmax(dim=1, keepdim=True)
            total_correct_task1 += pred1.eq(targets[0].view_as(pred1)).sum().item()

            # pred2 = outputs[1].argmax(dim=1, keepdim=True)
            # total_correct_task2 += pred2.eq(targets[1].view_as(pred2)).sum().item()

            # loss
            for task in range(args.num_tasks):
                test_losses[task] += losses[task]
    accuracy_task1 = total_correct_task1 / len(train_loader_test.dataset) * 100
    # accuracy_task2 = total_correct_task2 / len(train_loader_test.dataset) * 100

    # 计算每个任务的平均损失
    for task_idx in range(args.num_tasks):
        test_losses[task_idx] = test_losses[task_idx] / len(train_loader_test)
    print(
        'Client 0 Test - task1 correct:{:.2f}%'.format(accuracy_task1)
        # ,'task2 correct:{:.2f}%'.format(accuracy_task2)
    )
    print(
        "train1 loss:{:.4f}".format(test_losses[0].item())
        # ,"train2 loss:{:.4f}".format(test_losses[1].item())
    )
    print(".............................................")

    return (test_losses[0].item()
            # , test_losses[1].item()
            ), (accuracy_task1
                                                            # , accuracy_task2
                                                            )

if __name__ == "__main__":
    args = args_parser()
    # args.batch_size = 256
    torch.manual_seed(args.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.

    # 准备原始数据集
    # 不同分类生成一个批次
    train_dataset = generate_multi_mnist(num_samples=6000)

    # 生成测试数据
    test_dataset = generate_multi_mnist(num_samples=1000, train=False)

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
    loss_history = {'task1': {'256': []}
        # ,'task2': {'256': []}
                    }
    # loss_eval = {'task1': {'fmgda': [], 'fmgda_s': []}, 'task2': {'fmgda': [], 'fmgda_s': []}}
    # loss_eval = {'task1': {'16':[], '64':[], '128': [], '256': []}, 'task2': {'16':[], '64':[], '128': [], '256': []}}
    loss_eval = {'task1': {'256': []}
                    # ,'task2': {'256': []}
                 }
    # acc = {'task1': {'fmgda': [], 'fmgda_s': []}, 'task2': {'fmgda': [], 'fmgda_s': []}}
    # acc = {'task1': {'16':[], '64':[], '128': [], '256': []}, 'task2': {'16':[], '64':[], '128': [], '256': []}}
    acc = {'task1': {'256': []}
                    # ,'task2': {'256': []}
           }

    for batch_size in [256]:
        # 初始化模型参数
        server_model = ServerSharedModel()
        client_model = ClientMTLModel(server_model)
        clients_model = [copy.deepcopy(client_model) for _ in range(args.num_clients)]  # 模型初始化
        args.batch_size = batch_size

        # method == fmgda_s时对下列参数初始化
        last_clients_model = [copy.deepcopy(client_model) for _ in range(args.num_clients)]
        last_client_grads = last_client_init(client_model)

    # 开始训练
        for epoch in tqdm(range(1, args.global_epochs + 1)):
            print(f'\n | Global Training Round : {epoch + 1} |\n')
            if args.method == 'fmgda_s':
                server_model, clients_model, last_clients_model, last_client_grads,loss = train(args,
                                                                                           server_model=server_model,
                                                                                           clients_model=clients_model,
                                                                                           client_datasets=client_datasets,
                                                                                           method=args.method,
                                                                                           last_clients_model=last_clients_model,
                                                                                           last_client_grads=last_client_grads)
            else:
                server_model, clients_model, loss = train(args,
                                                    server_model=server_model,
                                                    clients_model=clients_model,
                                                    client_datasets=client_datasets,
                                                    method=args.method,
                                                    )
            test_loss, test_acc = test(args, clients_model, test_data=test_dataset)
            loss_eval['task1']["{}".format(batch_size)].append(test_loss)
            # loss_eval['task2']["{}".format(batch_size)].append(test_loss[1])
            acc['task1']["{}".format(batch_size)].append(test_acc)
            # acc['task2']["{}".format(batch_size)].append(test_acc[1])
            loss_history['task1']["{}".format(batch_size)].append(loss)
            # loss_history['task2']["{}".format(batch_size)].append(loss[1])

    # 保存模型参数
    torch.save(clients_model[0].state_dict(), '/kaggle/working/client0_model_model_params.pth')
    torch.save(server_model.state_dict(), '/kaggle/working/server_model_params.pth')

    # 数据保存
    # 将字典转换为扁平结构
    data1 = {}
    for task, methods in loss_history.items():
        for method, values in methods.items():
            col_name = f"{task}_{method}"  # 组合列名：任务_方法
            data1[col_name] = values

    df1 = pd.DataFrame(data1)
    df1.to_excel("/kaggle/working/loss_history.xlsx", index=False)
    #
    # # 将字典转换为扁平结构
    data2 = {}
    for task, methods in loss_eval.items():
        for method, values in methods.items():
            col_name = f"{task}_{method}"  # 组合列名：任务_方法
            data2[col_name] = values

    df2 = pd.DataFrame(data2)
    df2.to_excel("/kaggle/working/loss_eval.xlsx", index=False)
    #
    # # 将字典转换为扁平结构
    data3 = {}
    for task, methods in acc.items():
        for method, values in methods.items():
            col_name = f"{task}_{method}"  # 组合列名：任务_方法
            data3[col_name] = values

    df3 = pd.DataFrame(data3)
    df3.to_excel("/kaggle/working/acc.xlsx", index=False)

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
    plt.savefig('/kaggle/working/task1_mulit_loss_curve.png', dpi=300, bbox_inches='tight')

    # fig, ax = plt.subplots()
    # task2_loss = loss_history["task2"]
    # for i in [256]:
    #     ax.semilogy(range(len(task2_loss["{}".format(i)])), task2_loss["{}".format(i)], label="{}".format(i))
    # ax.spines['left'].set_position('zero')  # Y轴在x=0处
    # ax.set_xlim(left=0)  # 只显示x正半轴
    # ax.set_title("Task2 Train Loss")
    # ax.set_xlabel("Communication rounds")
    # ax.set_ylabel("Train Local Loss")
    # # ax.set_yticks([0.01, 0.1, max(max(task2_loss["128"]), max(task2_loss["256"]))])
    # ax.set_yticks([0.01, 0.1, max(task1_loss["256"])])
    # plt.legend()
    # plt.grid(False)
    # # 保存图像
    # plt.savefig('/kaggle/working/task2_mulit_loss_curve.png', dpi=300, bbox_inches='tight')

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
    plt.savefig('/kaggle/working/test_task1_mulit_loss_curve.png', dpi=300, bbox_inches='tight')

    # fig, ax = plt.subplots()
    # test_task2_loss = loss_eval["task2"]
    # for i in [256]:
    #     ax.semilogy(range(len(test_task2_loss["{}".format(i)])), test_task2_loss["{}".format(i)], label="{}".format(i))
    # ax.spines['left'].set_position('zero')  # Y轴在x=0处
    # ax.set_xlim(left=0)  # 只显示x正半轴
    # ax.set_title("Task2 Test Loss")
    # ax.set_xlabel("Communication rounds")
    # ax.set_ylabel("Test Local Loss")
    # # ax.set_yticks([0.01, 0.1, max(max(test_task2_loss["128"]), max(test_task2_loss["256"]))])
    # ax.set_yticks([0.01, 0.1, max(task1_loss["256"])])
    # plt.legend()
    # plt.grid(False)
    # # 保存图像
    # plt.savefig('/kaggle/working/test_task2_mulit_loss_curve.png', dpi=300, bbox_inches='tight')

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
    ax.set_yticks([1, 10, max(task1_acc["256"])])
    plt.legend()
    plt.grid(False)
    # 保存图像
    plt.savefig('/kaggle/working/task1_acc_curve.png', dpi=300, bbox_inches='tight')

    # fig, ax = plt.subplots()
    # task2_acc = acc["task2"]
    # for i in [256]:
    #     ax.plot(range(len(task2_acc["{}".format(i)])), task2_acc["{}".format(i)], label="{}".format(i))
    # ax.spines['left'].set_position('zero')  # Y轴在x=0处
    # ax.set_xlim(left=0)  # 只显示x正半轴
    # ax.set_title("Task2 Test Acc")
    # ax.set_xlabel("Communication rounds")
    # ax.set_ylabel("Test Acc")
    # ax.set_yticks([1, 10, max(task1_acc["256"])])
    # # ax.set_yticks([1, 10, max(task2_acc["fmgda"])])
    # plt.legend()
    # plt.grid(False)
    # # 保存图像
    # plt.savefig('/kaggle/working/task2_acc_curve.png', dpi=300, bbox_inches='tight')

    plt.show()