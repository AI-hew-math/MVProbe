import os
import json
import torch
import probex
import argparse
import shortuuid
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from safetensors.torch import load_model, save_file
from probex_datasets import ModelsDatasetDiscriminative
from utils import fix_seeds, str2bool, CIFAR100_ALL_CLASSES


def setup_training(args, device):
    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(CIFAR100_ALL_CLASSES))
    num_classes = len(label_encoder.classes_)
    assert num_classes == 100, f"Expected 100 classes, got {num_classes}"

    # Create datasets
    train_dataset = ModelsDatasetDiscriminative(
        os.path.join(args.input_path, 'train'),
        label_encoder,
        args.current_layer_name,
        num_classes=num_classes,
        is_resnet=args.is_resnet
    )
    val_dataset = ModelsDatasetDiscriminative(
        os.path.join(args.input_path, 'val'),
        label_encoder,
        args.current_layer_name,
        num_classes=num_classes,
        is_resnet=args.is_resnet
    )
    test_dataset = ModelsDatasetDiscriminative(
        os.path.join(args.input_path, 'test'),
        label_encoder,
        args.current_layer_name,
        num_classes=num_classes,
        is_resnet=args.is_resnet
    )

    # Create data loaders (seeded for reproducibility)
    num_workers = 4
    g = torch.Generator()
    g.manual_seed(args.seed)
    def worker_init_fn(worker_id):
        import numpy as np, random
        np.random.seed(args.seed + worker_id)
        random.seed(args.seed + worker_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn)

    args.layer_shape = train_dataset[0][0].shape

    model = probex.ProbeXClassification(
            input_shape=args.layer_shape, num_classes=num_classes,
            n_probes=args.n_probes, proj_dim=args.proj_dim, rep_dim=args.rep_dim,
            x_center=args.x_center, x_row_norm=args.x_row_norm,
        )
    model = model.to(device)

    if args.init_checkpoint:
        print(f"[INFO] Loading init checkpoint: {args.init_checkpoint}")
        load_model(model, args.init_checkpoint)
        model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.training_lr, weight_decay=1e-5)

    n_model_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTraining total number of trainable parameters: {n_trainable_params}/{n_model_params}")

    return model, train_loader, val_loader, test_loader, criterion, optimizer, n_model_params


def ablation_to_branches(ablation: str):
    if ablation in (None, '', 'none'):
        return None
    mapping = {
        'xu': ['xu'],
        'xtu': ['xtu'],
        'xxtu': ['xxtu'],
        'xtxu': ['xtxu'],
        'first_order': ['xu','xtu'],
        'second_order': ['xxtu','xtxu'],
    }
    return mapping.get(ablation, None)


def run_eval(args, model, eval_loader, criterion, device):
    model.eval()
    eval_loss = 0.0
    eval_correct = 0
    eval_total = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            active_branches = ablation_to_branches(args.ablation)
            outputs = model(inputs, active_branches=active_branches)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            threshold = 0.5
            predicted = (torch.sigmoid(outputs) > threshold).float()
            eval_total += labels.flatten().size(0)
            eval_correct += (predicted == labels).sum().item()

    eval_loss /= len(eval_loader)
    eval_accuracy = eval_correct / eval_total
    return eval_loss, eval_accuracy


def run_train(args, model, train_loader, val_loader, criterion, optimizer, device):
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    best_val_epoch = float('inf')
    best_epoch_train_loss = float('inf')
    best_epoch_train_accuracy = 0.0

    # Ablation: branch subset masking
    active_branches = ablation_to_branches(args.ablation)

    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs, active_branches=active_branches)

            # classification loss
            loss_cls = criterion(outputs, labels)
            loss = loss_cls

            loss.backward()
            optimizer.step()

            train_loss += loss_cls.item()
            with torch.no_grad():
                threshold = 0.5
                predicted = (torch.sigmoid(outputs) > threshold).float()
                train_total += labels.flatten().size(0)
                train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        print(f"\nEpoch [{epoch + 1}/{args.n_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {(100 * train_accuracy):.2f}%")

        if epoch % 25 == 0:
            val_loss, val_accuracy = run_eval(args, model, val_loader, criterion, device=device)
            print(f"\nEpoch [{epoch + 1}/{args.n_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {(100 * val_accuracy):.2f}%")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                best_val_epoch = epoch
                best_epoch_train_loss = train_loss
                best_epoch_train_accuracy = train_accuracy
                save_file(model.state_dict(), args.checkpoint_save_path, metadata={"epoch": str(epoch)})

    return best_epoch_train_loss, best_epoch_train_accuracy, best_val_loss, best_val_accuracy, best_val_epoch


def train_layer(args):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    model, train_loader, val_loader, test_loader, criterion, optimizer, n_model_params = setup_training(args, device)
    train_loss, train_acc, val_loss, val_acc, best_epoch = run_train(
        args, model, train_loader, val_loader, criterion, optimizer, device=device
    )

    load_model(model, args.training_checkpoint_path)
    model = model.to(device)

    test_loss, test_acc = run_eval(args, model, test_loader, criterion, device=device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {(100 * test_acc):.2f}%")

    # Cleanup: delete checkpoint file after test
    if args.cleanup_checkpoints:
        try:
            if os.path.isfile(args.training_checkpoint_path):
                os.remove(args.training_checkpoint_path)
        except Exception as e:
            print(f"[WARN] Failed to remove checkpoint {args.training_checkpoint_path}: {e}")

    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, n_model_params, best_epoch


def default_argument_parser():
    parser = argparse.ArgumentParser(description='Train ProbeX or Dense for ViT models')
    parser.add_argument('--input_path', type=str, help='The path to the training dataset', required=True)
    parser.add_argument('--output_path', type=str, help='The output path', required=True)
    parser.add_argument('--is_resnet', type=str2bool, help='Whether to the dataset is of ResNets or ViTs', required=True)
    parser.add_argument('--suffix', type=str, help='The suffix to use', default='')

    parser.add_argument('--start_layer', type=int, help='The index of the first layer to classify', default=0)
    parser.add_argument('--n_layers', type=int, help='The number of layers to classify', default=1)

    parser.add_argument('--proj_dim', type=int, help='The dimension to project the weight matrix to', default=128)
    parser.add_argument('--n_probes', type=int, help='The number of probes to use', default=128)
    parser.add_argument('--rep_dim', type=int, help='The dimension of the final representation', default=512)

    # Ablation control (branch subset masking)
    parser.add_argument(
        '--ablation', type=str, default='none',
        choices=['none','xu','xtu','xxtu','xtxu','first_order','second_order'],
        help=(
            "Branch subset masking on top of concat_all4_bigenc_*: "
            "none(all), xu, xtu, xxtu, xtxu, first_order(xu+xtu), second_order(xxtu+xtxu)."
        ),
    )
    parser.add_argument('--x_center', type=str2bool, default=False,
                        help="Center the point cloud used for probing by subtracting mean over points.")
    parser.add_argument('--x_row_norm', type=str2bool, default=False,
                        help="L2-normalize each point before probing.")

    parser.add_argument('--training_lr', type=float, help='The learning rate for the model', default=3e-4)
    parser.add_argument('--batch_size', type=int, help='The batch size for the model', default=128)
    parser.add_argument('--n_epochs', type=int, help='The number of epochs for the model', default=3000)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--init_checkpoint', type=str, default='',
                        help="(Optional) Checkpoint path to load before training.")

    parser.add_argument('--gpu_id', '-gpu', type=int, default=0, help='GPU index to use, e.g. 0 or 1')

    parser.add_argument('--cleanup_checkpoints', type=str2bool, default=False,
                        help="After test eval, delete best checkpoint file to save disk.")

    return parser


def setup_paths(args):
    args.rep_dim_out = args.rep_dim

    args.output_path_root = f"{args.output_path}/ProbeX/abl-{args.ablation}/n_probes-{args.n_probes}/proj_dim-{args.proj_dim}/rep_dim-{args.rep_dim_out}"
    os.makedirs(args.output_path_root, exist_ok=True)
    suffix = args.suffix if args.suffix else shortuuid.ShortUUID().random(length=5)
    args.final_suffix = f"{suffix}_{datetime.now().strftime('%H-%M-%d-%m')}"
    os.makedirs(os.path.join(args.output_path_root, "args"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path_root, "checkpoints", "training"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path_root, "results"), exist_ok=True)


def main():
    args = default_argument_parser().parse_args()
    if args.is_resnet:
        all_layer_names = pd.read_csv("./layer_names/resnet_weight_indices_to_layer_names.csv")['0'].tolist()
    else:
        all_layer_names = pd.read_csv("./layer_names/vit_weight_indices_to_layer_names.csv")['0'].tolist()

    setup_paths(args)
    fix_seeds(args)

    print(f"Running the model with the following arguments: {json.dumps(vars(args), indent=4)}")

    for layer_idx in tqdm(range(args.start_layer, args.start_layer + args.n_layers)):
        args.current_layer = layer_idx
        args.current_layer_name = all_layer_names[args.current_layer]
        args.training_checkpoint_path = os.path.join(
            args.output_path_root, "checkpoints", "training",
            f"training_best_model__layer-{args.current_layer}__{args.final_suffix}.safetensors"
        )
        with open(os.path.join(args.output_path_root, "args", f"args_layer-{args.current_layer}__{args.final_suffix}.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        print("\n\n\n" + "=" * 25 + " NEW LAYER " + "=" * 25)
        print(f"Starting to run on layer {args.current_layer} (layer name: {args.current_layer_name})")

        args.checkpoint_save_path = args.training_checkpoint_path
        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, n_model_params, best_epoch = train_layer(args)

        layer_results = {args.current_layer: {
            "layer_idx": args.current_layer,
            "layer_name": args.current_layer_name,
            "ablation": args.ablation,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "number_of_params": n_model_params,
            "best_epoch": best_epoch,
            "layer_shape": args.layer_shape,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "training_checkpoint_path": args.training_checkpoint_path,
            "n_probes": args.n_probes,
            "proj_dim": args.proj_dim,
            "rep_dim_branch": args.rep_dim,
            "rep_dim_out": getattr(args, "rep_dim_out", args.rep_dim),
            "batch_size": args.batch_size,
        }}

        results_path = os.path.join(args.output_path_root, "results", f"results_layer-{args.current_layer}__{args.final_suffix}.csv")
        pd.DataFrame(layer_results.values()).to_csv(results_path, index=False)
        print(f"Finished layer {args.current_layer} (layer name: {args.current_layer_name}), results saved to {results_path}")


if __name__ == "__main__":
    main()
