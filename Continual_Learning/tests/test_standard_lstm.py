import unittest
import sys
import os
import numpy as np
import torch

import torch.nn as nn
from torch import optim

from libauc.losses import pAUC_DRO_Loss
from libauc.optimizers import SOPAs
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

from torchmimic.benchmarks import (
    IHMBenchmark,
    DecompensationBenchmark,
    LOSBenchmark,
    PhenotypingBenchmark,
)

from torchmimic.loggers import (
    IHMLogger,
    DecompensationLogger,
    LOSLogger,
    PhenotypingLogger,
)

from torchmimic.models import StandardLSTM
from torchmimic.utils import (
    get_test_loaders,
    get_val_loaders,
    get_train_loader,
    update_buffer,
    get_samples,
)

lf_map = ["south", "midwest", "west", "northeast"]

# Server Paths
ihm_tasks = [
    "/data/datasets/mimic3-benchmarks/data/in-hospital-mortality",
    "/data/datasets/eICU2MIMIC/ihm",
]
ihm_splits = [
    "/data/datasets/mimic3-benchmarks/data/in-hospital-mortality",
    "/data/datasets/eICU2MIMIC/ihm_split",
    "/data/datasets/eICU2MIMIC/ihm_split",
    "/data/datasets/eICU2MIMIC/ihm_split",
    "/data/datasets/eICU2MIMIC/ihm_split",
]

phen_tasks = [
    "/data/datasets/mimic3-benchmarks/data/phenotyping",
    "/data/datasets/eICU2MIMIC/phenotyping",
]
phen_splits = [
    "/data/datasets/mimic3-benchmarks/data/phenotyping",
    "/data/datasets/eICU2MIMIC/phenotyping_split",
    "/data/datasets/eICU2MIMIC/phenotyping_split",
    "/data/datasets/eICU2MIMIC/phenotyping_split",
    "/data/datasets/eICU2MIMIC/phenotyping_split",
]

los_tasks = [
    "/data/datasets/mimic3-benchmarks/data/length-of-stay",
    "/data/datasets/eICU2MIMIC/length-of-stay",
]
los_splits = [
    "/data/datasets/mimic3-benchmarks/data/length-of-stay",
    "/data/datasets/eICU2MIMIC/length-of-stay_split",
    "/data/datasets/eICU2MIMIC/length-of-stay_split",
    "/data/datasets/eICU2MIMIC/length-of-stay_split",
    "/data/datasets/eICU2MIMIC/length-of-stay_split",
]

decomp_tasks = [
    "/data/datasets/mimic3-benchmarks/data/decompensation",
    "/data/datasets/eICU2MIMIC/decompensation",
]
decomp_splits = [
    "/data/datasets/mimic3-benchmarks/data/decompensation",
    "/data/datasets/eICU2MIMIC/decompensation_split",
    "/data/datasets/eICU2MIMIC/decompensation_split",
    "/data/datasets/eICU2MIMIC/decompensation_split",
    "/data/datasets/eICU2MIMIC/decompensation_split",
]

# Local paths
# ihm_tasks = [
#     "../../datasets/mimic3-benchmarks/in-hospital-mortality",
#     "../../datasets/eICU-benchmarks/data_mimicformat/in-hospital-mortality2",
# ]
# ihm_splits = [
#     "../../datasets/mimic3-benchmarks/in-hospital-mortality",
#     "/data/datasets/eICU2MIMIC/ihm_split",
#     "/data/datasets/eICU2MIMIC/ihm_split",
#     "/data/datasets/eICU2MIMIC/ihm_split",
#     "/data/datasets/eICU2MIMIC/ihm_split",
# ]

# phen_tasks = [
#     "../../datasets/mimic3-benchmarks/phenotyping",
#     "../../datasets/eICU-benchmarks/data_mimicformat/phenotyping",
# ]
# phen_splits = [
#     "../../datasets/mimic3-benchmarks/phenotyping",
#     "/data/datasets/eICU2MIMIC/phenotyping_split",
#     "/data/datasets/eICU2MIMIC/phenotyping_split",
#     "/data/datasets/eICU2MIMIC/phenotyping_split",
#     "/data/datasets/eICU2MIMIC/phenotyping_split",
# ]

# los_tasks = [
#     "../../datasets/mimic3-benchmarks/length-of-stay",
#     "../../datasets/eICU-benchmarks/data_mimicformat/length-of-stay",
# ]
# los_splits = [
#     "../../datasets/mimic3-benchmarks/length-of-stay",
#     "/data/datasets/eICU2MIMIC/length-of-stay_split",
#     "/data/datasets/eICU2MIMIC/length-of-stay_split",
#     "/data/datasets/eICU2MIMIC/length-of-stay_split",
#     "/data/datasets/eICU2MIMIC/length-of-stay_split",
# ]

# decomp_tasks = [
#     "../../datasets/mimic3-benchmarks/decompensation",
#     "../../datasets/eICU-benchmarks/data_mimicformat/decompensation",
# ]
# decomp_splits = [
#     "../../datasets/mimic3-benchmarks/decompensation",
#     "/data/datasets/eICU2MIMIC/decompensation_split",
#     "/data/datasets/eICU2MIMIC/decompensation_split",
#     "/data/datasets/eICU2MIMIC/decompensation_split",
#     "/data/datasets/eICU2MIMIC/decompensation_split",
# ]


def get_config(
    test_batch_size,
    train_batch_size,
    learning_rate,
    weight_decay,
    ewc_penalty=False,
    importance=0,
    replay=False,
    buffer_size=0,
    epochs=2,
    tasks=1,
):
    return {
        "Buffer size": buffer_size,
        "EWC": ewc_penalty,
        "Importance": importance,
        "Replay": replay,
        "test_batch_size": test_batch_size,
        "train_batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "Epochs": epochs,
        "Tasks": tasks,
    }


class TestLSTM(unittest.TestCase):
    def test_standard_lstm(
        self,
        task,
        epochs=2,
        task_list=[0],
        buffer_size=0,
        replay=False,
        rpl_type="adjrep",
        ewc_penalty=False,
        importance=0,
        test=False,
        pAUC=False,
        region=0,
        bilstm=True,
    ):

        device = 0
        sample_size = None  # use all samples (IHM & Phen only)
        train_batch_size = 8
        test_batch_size = 256
        learning_rate = 0.001
        weight_decay = 0
        report_freq = 200
        workers = 5

        wandb = True
        config = {}
        test_loaders = None
        exp_name = task
        exp_name += f"_{lf_map[region-1]}" if region != 0 else ""
        if buffer_size == 0:
            ewc_penalty = False
            replay = False
            exp_name += "_baseline"
            print("NOTE: Buffer size is 0, EWC and Replay will not be used")

        # specify tasks
        if task == "ihm":
            if region == 0:
                tasks = (
                    [ihm_tasks[i] for i in task_list]
                    if len(task_list) <= 2
                    else [ihm_splits[i] for i in task_list]
                )
            else:
                tasks = [ihm_splits[0], ihm_splits[1]]
            model = StandardLSTM(
                n_classes=1,
                hidden_dim=16,
                num_layers=2,
                dropout_rate=0.3,
                bidirectional=bilstm,
            )
            logger = IHMLogger
            benchmark = IHMBenchmark
            # For use in pAUC
            shift_map = [0]
            data_len = 62475  # Total number of IHM Samples
            crit = nn.BCELoss()

        elif task == "phen":
            if region == 0:
                tasks = (
                    [phen_tasks[i] for i in task_list]
                    if len(task_list) <= 2
                    else [phen_splits[i] for i in task_list]
                )
            else:
                tasks = [phen_splits[0], phen_splits[1]]
            model = StandardLSTM(
                n_classes=25,
                hidden_dim=256,
                num_layers=1,
                dropout_rate=0.3,
                bidirectional=bilstm,
            )
            logger = PhenotypingLogger
            benchmark = PhenotypingBenchmark
            # For use in pAUC
            shift_map = [0]
            data_len = 60891  # Total number of Pheno Samples
            crit = nn.BCELoss()

        elif task == "decomp":
            # use 100k for first three tasks, then 50k for task four, and 25k for task 5
            sample_size = 100000
            if region == 0:
                tasks = (
                    [decomp_tasks[i] for i in task_list]
                    if len(task_list) <= 2
                    else [decomp_splits[i] for i in task_list]
                )
            else:
                tasks = [decomp_splits[0], decomp_splits[1]]
            model = StandardLSTM(
                n_classes=1,
                hidden_dim=128,
                num_layers=1,
                dropout_rate=0.3,
                bidirectional=bilstm,
            )
            logger = DecompensationLogger
            benchmark = DecompensationBenchmark
            # For use in pAUC
            shift_map = [0]
            data_len = np.sum(
                [x * sample_size for x in [1, 1, 1, 0.5, 0.25]]
            )  # 1, 1, 1, .5 .25 sample ratios
            crit = nn.BCELoss()

        elif task == "los":
            # use 100k for first three tasks, then 50k for task four, and 25k for task 5
            sample_size = 100000
            if region == 0:
                tasks = (
                    [los_tasks[i] for i in task_list]
                    if len(task_list) <= 2
                    else [los_splits[i] for i in task_list]
                )
            else:
                tasks = [los_splits[0], los_splits[1]]
            model = StandardLSTM(
                n_classes=10,
                hidden_dim=64,
                num_layers=1,
                dropout_rate=0.3,
                bidirectional=bilstm,
            )
            logger = LOSLogger
            benchmark = LOSBenchmark
            # For use in pAUC
            shift_map = [0]
            data_len = np.sum(
                [x * sample_size for x in [1, 1, 1, 0.5, 0.25]]
            )  # 1, 1, 1, .5 .25 sample ratios
            crit = nn.CrossEntropyLoss()

        config.update(model.get_config())
        config.update(
            get_config(
                test_batch_size,
                train_batch_size,
                learning_rate,
                weight_decay,
                ewc_penalty,
                importance,
                replay,
                buffer_size,
                epochs,
                len(tasks),
            )
        )

        exp_name = ("Test_" + exp_name) if test else ("Train_" + exp_name)
        logger = logger(exp_name, config, wandb)

        # get test loaders for each task
        val_loaders = get_val_loaders(
            task, tasks, lf_map, test_batch_size, sample_size, workers, device, region
        )

        if test:
            test_loaders = get_test_loaders(
                task,
                tasks,
                lf_map,
                test_batch_size,
                sample_size,
                workers,
                device,
                region,
            )

        # train on current task, test on all tasks
        prev_model = None
        samples = {}
        val_results = []
        test_results = {}
        buffer = []
        shift = 0

        if pAUC:
            crit = pAUC_DRO_Loss(data_len=int(data_len))
            optimizer = SOPAs(  # initial optimizer
                model.parameters(),
                mode="adam",
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.98),
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.98),
            )

        for task_num, task_data in enumerate(tasks):
            # use model from previous trainer for additional tasks
            if task_num > 0:
                model = prev_model
                optimizer = (
                    SOPAs(  # update model parameters
                        model.parameters(),
                        mode="adam",
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        betas=(0.9, 0.98),
                    )
                    if pAUC
                    else optim.Adam(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        betas=(0.9, 0.98),
                    )
                )

            trainer = benchmark(
                model=model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                report_freq=report_freq,
                logger=logger,
                device=device,
                loss=crit,
                optimizer=optimizer,
                shift_map=shift_map,
                pAUC=pAUC,
            )
            # train model, evaluate on all testing data
            train_loader = get_train_loader(
                task_num,
                task,
                tasks,
                lf_map,
                train_batch_size,
                sample_size,
                workers,
                device,
                pAUC,
                region,
            )
            result = trainer.fit(
                epochs,
                train_loader,
                val_loaders,
                test_loaders,
                task_num,
                buffer,
                replay=replay,
                rpl_type=rpl_type,
                ewc_penalty=ewc_penalty,
                importance=importance,
            )

            shift += len(train_loader.dataset)  # keep track of index shift for pAUC
            shift_map.append(shift)
            if test:
                test_results["Task " + str(task_num + 1)] = result["test"]
            val_results.append(result["val"])

            # get random samples for ewc/replay
            if (task_num != len(tasks) - 1) and (ewc_penalty or replay):
                samples["Task " + str(task_num + 1)] = get_samples(
                    task_num, buffer_size, train_loader
                )
                buffer = update_buffer(task_num, samples, buffer_size)
            prev_model = trainer.model
            del trainer, train_loader

        m1, m2 = logger.update_wandb_val(val_results)
        results = {}
        results["val"] = ((m1, m2), logger.get_val_scores(), config)

        if test:
            logger.update_wandb_test(test_results)
            results["test"] = ((m1, m2), logger.get_test_scores(), config)
            sources = get_conf_matrix_stats(task, model, test_loaders, device, region)
            results["conf_matrix"] = sources
            # is_float = np.vectorize(lambda x: isinstance(x, float))(sources)
            # if np.all(is_float):
            #     print("------")
            #     print("All elements are floats.")
            #     print(results["conf_matrix"])
            # else:
            #     print("------")
            #     print("Not all elements are floats.")
            #     print(results["conf_matrix"])
        return results


def get_conf_matrix_stats(task, model, test_loaders, device, region):
    model.eval()
    sources = []
    with torch.no_grad():
        for source, test_loader in enumerate(test_loaders):
            y_true = []
            y_pred = []
            for batch_idx, (data, label, lens, mask, index) in enumerate(test_loader):
                data = data.to(device)
                label = label.to(device)
                outputs = model((data, lens))
                if task == "los":
                    _, predicted = torch.max(outputs, 1)
                else:
                    threshold = 0.5
                    predicted = (outputs > threshold).long()

                y_true.extend(label.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

            if task == "phen":
                report = classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                    labels=list(range(25)),
                )
                print(report)
                sensitivity = [report[str(i)]["recall"] for i in range(25)]
                specificity = multilabel_specificity(y_true, y_pred)
                assert len(sensitivity) == 25 and len(specificity) == 25
                sources.append([sensitivity, specificity])
            elif task == "los":
                report = classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                    labels=list(range(10)),
                )
                print(report)
                sensitivity = [report[str(i)]["recall"] for i in range(10)]
                specificity = multiclass_specificity(y_true, y_pred)
                assert len(sensitivity) == 10 and len(specificity) == 10
                sources.append([sensitivity, specificity])
            else:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                sources.append([sensitivity, specificity])
        return np.array(sources)


def multilabel_specificity(y_true, y_pred, num_labels=25):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    specificity_scores = []
    for label in range(num_labels):
        true_label = y_true[:, label]
        pred_label = y_pred[:, label]
        tn, fp, fn, tp = confusion_matrix(true_label, pred_label).ravel()

        specificity = tn / (tn + fp)
        specificity_scores.append(specificity)

    return specificity_scores


def multiclass_specificity(y_true, y_pred, num_classes=10):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    specificity_scores = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        specificity = tn / (tn + fp)
        specificity_scores.append(specificity)

    return specificity_scores
