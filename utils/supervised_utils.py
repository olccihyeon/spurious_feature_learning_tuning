import utils
from utils import AverageMeter
import tqdm
import torch
import wandb
import torch.nn.functional as F
import torch.nn as nn
import os


class CustomModel(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomModel, self).__init__()
        self.model = pretrained_model

        # 마지막 계층에 Sigmoid 활성화 함수 추가
        self.sigmoid = nn.Sigmoid()
        self.fc = self.model.fc

    def forward(self, x):
        out = self.model(x)
        return self.sigmoid(out)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_meter = AverageMeter()
    n_groups = loader.dataset.n_groups
    acc_groups = {g_idx: AverageMeter() for g_idx in range(n_groups)}

    for batch in (pbar := tqdm.tqdm(loader)):
        x, y, g, s = batch
        x, y, s = x.cuda(), y.cuda(), s.cuda()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss, x.size(0))

        preds = torch.argmax(logits, dim=1)
        if len(y.shape) > 1:
            # mixup
            y = torch.argmax(y, dim=1)

        utils.update_dict(acc_groups, y, g, logits)
        acc = (preds == y).float().mean()

        pbar.set_description("Loss: {:.3f} ({:3f}); Acc: {:3f}".format(loss.item(), loss_meter.avg, acc))

    return loss_meter, acc_groups, x


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred.squeeze()


def dfr_train_epoch(model, loader, criterion):
    model.eval()
    model_train = LogisticRegression(2048, 1).cuda()
    for name, param in model_train.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    print("criterion : ", criterion)
    model_train.train()
    optimizer = torch.optim.AdamW(model_train.parameters(), lr=0.0001, weight_decay=0.001)
    x_concat_tensor = torch.tensor([]).cuda()
    y_concat_tensor = torch.tensor([]).cuda()
    g_concat_tensor = torch.tensor([])
    p_concat_tensor = torch.tensor([]).cuda()
    for batch in (pbar := tqdm.tqdm(loader)):
        x, y, g, p = batch
        x, y, p = x.cuda(), y.cuda(), p.cuda()
        x_embed = model(x)

        x_concat_tensor = torch.cat((x_concat_tensor, x_embed), 0)
        y_concat_tensor = torch.cat((y_concat_tensor, y), 0)
        g_concat_tensor = torch.cat((g_concat_tensor, g), 0)
        p_concat_tensor = torch.cat((p_concat_tensor, p), 0)
    for i in range(500):
        n_groups = int(torch.max(g_concat_tensor).item()) + 1

        g_idx = [torch.where(g_concat_tensor == g)[0] for g in range(n_groups)]
        min_g = min([len(g) for g in g_idx])

        # 무작위 셔플
        for g in g_idx:
            g = g[torch.randperm(len(g))]

        # 텐서 x_train, y_train, g_train 생성
        x_train = torch.cat([x_concat_tensor[g[:min_g]] for g in g_idx], dim=0)
        y_train = torch.cat([y_concat_tensor[g[:min_g]] for g in g_idx], dim=0)
        g_train = torch.cat([g_concat_tensor[g[:min_g]] for g in g_idx], dim=0)

        logits = model_train(x_train)
        optimizer.zero_grad()
        loss = criterion(logits, y_train)
        # print("loss:", loss.item())

        loss.backward()
        optimizer.step()
        preds_val = torch.round(model_train(x_concat_tensor))
        n_groups = int(torch.max(g_concat_tensor).item()) + 1

        # print("val_mean_acc:", val_mean_acc.item(), "val_worst_acc:", val_worst_acc.item())

    preds_val = torch.round(model_train(x_concat_tensor))
    n_groups = int(torch.max(g_concat_tensor).item()) + 1
    val_accs = [(preds_val == y_concat_tensor)[g_concat_tensor == g].float().mean() for g in range(n_groups)]
    val_mean_acc = torch.mean((preds_val == y_concat_tensor).float())
    val_worst_acc = torch.min(torch.tensor(val_accs, dtype=torch.float))

    torch.save(model_train.state_dict(), "model_train4.pt")

    return val_accs, val_mean_acc, val_worst_acc


def train_epoch_DFR_fg_indirect(feature_extractor, linear_layer, loader, loader_bg, criterion, tuning, args):
    feature_extractor.eval()
    if tuning == "retraining":
        linear_layer = nn.Linear(2048, 2).cuda()
    elif tuning == "finetuning":
        linear_layer = nn.Linear(2048, 2).cuda()
        linear_layer.load_state_dict(
            torch.load(os.path.join(args.output_dir, f"linear_layer_seed{args.seed}_{args.DFR_data}.pt"))
        )

    linear_layer.train()
    optimizer = torch.optim.AdamW(linear_layer.parameters(), lr=0.001, weight_decay=0.001)

    x_concat_tensor = torch.tensor([]).cuda()
    y_concat_tensor = torch.tensor([]).cuda()
    g_concat_tensor = torch.tensor([])
    p_concat_tensor = torch.tensor([]).cuda()
    for batch in (pbar := tqdm.tqdm(loader)):
        x, y, g, p = batch
        x, y, p = x.cuda(), y.cuda(), p.cuda()
        x_embed = feature_extractor(x)

        x_concat_tensor = torch.cat((x_concat_tensor, x_embed), 0)
        y_concat_tensor = torch.cat((y_concat_tensor, y), 0)
        g_concat_tensor = torch.cat((g_concat_tensor, g), 0)
        p_concat_tensor = torch.cat((p_concat_tensor, p), 0)

    x_concat_tensor_bg = torch.tensor([]).cuda()
    for batch in (pbar := tqdm.tqdm(loader_bg)):
        x, y, g, p = batch
        x = x.cuda()
        x_embed = feature_extractor(x)
        x_concat_tensor_bg = torch.cat((x_concat_tensor_bg, x_embed), 0)

    x_concat_tensor = x_concat_tensor - x_concat_tensor_bg

    for i in range(20):
        n_groups = int(torch.max(g_concat_tensor).item()) + 1

        g_idx = [torch.where(g_concat_tensor == g)[0] for g in range(n_groups)]
        min_g = min([len(g) for g in g_idx])

        # 무작위 셔플
        for g in g_idx:
            g = g[torch.randperm(len(g))]

        # 텐서 x_train, y_train, g_train 생성
        x_train = torch.cat([x_concat_tensor[g[:min_g]] for g in g_idx], dim=0)
        y_train = torch.cat([y_concat_tensor[g[:min_g]] for g in g_idx], dim=0).long()
        g_train = torch.cat([g_concat_tensor[g[:min_g]] for g in g_idx], dim=0)

        optimizer.zero_grad()
        logits = linear_layer(x_train)
        loss = criterion(logits, y_train)
        # print("loss:", loss.item())
        # loss.requires_grad = True
        # print("loss:", loss.item())

        loss.backward()
        optimizer.step()

    preds_val = torch.argmax(linear_layer(x_concat_tensor), dim=1)
    n_groups = int(torch.max(g_concat_tensor).item()) + 1
    val_accs = [(preds_val == y_concat_tensor)[g_concat_tensor == g].float().mean() for g in range(n_groups)]
    val_mean_acc = torch.mean((preds_val == y_concat_tensor).float())
    val_worst_acc = torch.min(torch.tensor(val_accs, dtype=torch.float))

    torch.save(
        linear_layer.state_dict(),
        os.path.join(args.output_dir, f"linear_layer_seed{args.seed}_{tuning}_{args.DFR_data}.pt"),
    )

    return val_accs, val_mean_acc, val_worst_acc


def train_epoch_DFR(feature_extractor, linear_layer, loader, criterion, tuning, args):
    feature_extractor.eval()
    if tuning == "retraining":
        linear_layer = nn.Linear(2048, 2).cuda()
    elif tuning == "finetuning":
        linear_layer = nn.Linear(2048, 2).cuda()
        linear_layer.load_state_dict(
            torch.load(os.path.join(args.output_dir, f"linear_layer_seed{args.seed}_{args.DFR_data}.pt"))
        )

    linear_layer.train()
    optimizer = torch.optim.AdamW(linear_layer.parameters(), lr=0.001, weight_decay=0.001)

    x_concat_tensor = torch.tensor([]).cuda()
    y_concat_tensor = torch.tensor([]).cuda()
    g_concat_tensor = torch.tensor([])
    p_concat_tensor = torch.tensor([]).cuda()
    for batch in (pbar := tqdm.tqdm(loader)):
        x, y, g, p = batch
        x, y, p = x.cuda(), y.cuda(), p.cuda()
        x_embed = feature_extractor(x)

        x_concat_tensor = torch.cat((x_concat_tensor, x_embed), 0)
        y_concat_tensor = torch.cat((y_concat_tensor, y), 0)
        g_concat_tensor = torch.cat((g_concat_tensor, g), 0)
        p_concat_tensor = torch.cat((p_concat_tensor, p), 0)

    for i in range(20):
        n_groups = int(torch.max(g_concat_tensor).item()) + 1

        g_idx = [torch.where(g_concat_tensor == g)[0] for g in range(n_groups)]
        min_g = min([len(g) for g in g_idx])

        # 무작위 셔플
        for g in g_idx:
            g = g[torch.randperm(len(g))]

        # 텐서 x_train, y_train, g_train 생성
        x_train = torch.cat([x_concat_tensor[g[:min_g]] for g in g_idx], dim=0)
        y_train = torch.cat([y_concat_tensor[g[:min_g]] for g in g_idx], dim=0).long()
        g_train = torch.cat([g_concat_tensor[g[:min_g]] for g in g_idx], dim=0)

        optimizer.zero_grad()
        logits = linear_layer(x_train)
        loss = criterion(logits, y_train)

        loss.backward()
        optimizer.step()

    preds_val = torch.argmax(linear_layer(x_concat_tensor), dim=1)
    n_groups = int(torch.max(g_concat_tensor).item()) + 1
    val_accs = [(preds_val == y_concat_tensor)[g_concat_tensor == g].float().mean() for g in range(n_groups)]
    val_mean_acc = torch.mean((preds_val == y_concat_tensor).float())
    val_worst_acc = torch.min(torch.tensor(val_accs, dtype=torch.float))

    torch.save(
        linear_layer.state_dict(),
        os.path.join(args.output_dir, f"linear_layer_seed{args.seed}_{tuning}_{args.DFR_data}.pt"),
    )

    return val_accs, val_mean_acc, val_worst_acc


def eval(model, test_loader_dict):
    model.eval()
    results_dict = {}
    with torch.no_grad():
        # Currently test_loader_dict has "test" and "val"
        for test_name, test_loader in test_loader_dict.items():
            acc_groups = {g_idx: AverageMeter() for g_idx in range(test_loader.dataset.n_groups)}
            for x, y, g, p in tqdm.tqdm(test_loader):
                x, y, p = x.cuda(), y.cuda(), p.cuda()
                logits = model(x)
                utils.update_dict(acc_groups, y, g, logits)
            results_dict[test_name] = acc_groups
    return results_dict


def eval_DFR_pure(feature_extractor, linear_layer, test_loader_dict, tuning, args):
    linear_layer.load_state_dict(
        torch.load(os.path.join(args.output_dir, f"linear_layer_seed{args.seed}_{tuning}_{args.DFR_data}.pt"))
    )
    linear_layer.eval()
    feature_extractor.eval()
    results_dict = {}
    with torch.no_grad():
        # Currently test_loader_dict has "test" and "val"
        for test_name, test_loader in test_loader_dict.items():
            acc_groups = {g_idx: AverageMeter() for g_idx in range(test_loader.dataset.n_groups)}
            for x, y, g, p in tqdm.tqdm(test_loader):
                x, y, p = x.cuda(), y.cuda(), p.cuda()
                logits = linear_layer(feature_extractor(x))
                utils.update_dict(acc_groups, y, g, logits)
            results_dict[test_name] = acc_groups
    return results_dict


def dfr_eval(
    model,
    loader,
):
    model.eval()
    checkpoint = torch.load("model_train4.pt")
    model_train = LogisticRegression(2048, 1).cuda()
    model_train.load_state_dict(checkpoint)
    model_train.eval()
    results_dict = {}
    x_test = torch.tensor([]).cuda()
    y_test = torch.tensor([]).cuda()
    g_test = torch.tensor([])

    for batch in (pbar := tqdm.tqdm(loader)):
        x, y, g, p = batch
        x, y, p = x.cuda(), y.cuda(), p.cuda()
        x_embed = model(x)

        x_test = torch.cat((x_test, x_embed), 0)
        y_test = torch.cat((y_test, y), 0)
        g_test = torch.cat((g_test, g), 0)

    preds_test = model_train(x_test)
    preds_test = torch.round(preds_test)

    n_groups = int(torch.max(g_test).item()) + 1
    test_accs = [(preds_test == y_test)[g_test == g].float().mean() for g in range(n_groups)]
    test_mean_acc = torch.mean((preds_test == y_test).float())
    test_worts_acc = torch.min(torch.tensor(test_accs, dtype=torch.float))
    return test_accs, test_mean_acc, test_worts_acc


def eval_DFR(
    feature_extractor,
    linear_layer,
    loader,
    tuning,
):
    feature_extractor.eval()
    linear_layer.load_state_dict(torch.load(f"linear_layer_{tuning}.pt"))
    linear_layer.eval()
    results_dict = {}
    x_test = torch.tensor([]).cuda()
    y_test = torch.tensor([]).cuda()
    g_test = torch.tensor([])

    for batch in (pbar := tqdm.tqdm(loader)):
        x, y, g, p = batch
        x, y, p = x.cuda(), y.cuda(), p.cuda()
        x_embed = feature_extractor(x)

        x_test = torch.cat((x_test, x_embed), 0)
        y_test = torch.cat((y_test, y), 0)
        g_test = torch.cat((g_test, g), 0)

    preds_test = torch.argmax(linear_layer(x_test), dim=1)

    n_groups = int(torch.max(g_test).item()) + 1
    test_accs = [(preds_test == y_test)[g_test == g].float().mean() for g in range(n_groups)]
    test_mean_acc = torch.mean((preds_test == y_test).float())
    test_worts_acc = torch.min(torch.tensor(test_accs, dtype=torch.float))
    return test_accs, test_mean_acc, test_worts_acc
