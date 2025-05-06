#!/usr/bin/env python3
import logging
import os
import sys

import speechbrain as sb
import torch
# import torchaudio
import librosa
from hyperpyyaml import load_hyperpyyaml
import math

"""Recipe for training an Accent Identification (AID) system GenAID, with CommonAccent dataset.

To run this recipe, do the following:
> python train_GenAID.py train_GenAID_v6.yaml

Authors
------
 * Created by Juan Pablo Zuluaga 2023
 * Modified by Jinzuomu Zhong 2024
"""

logger = logging.getLogger(__name__)

# Brain class for Accent ID training
class AID(sb.Brain):
    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, wav_lens = wavs
        
        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN and hparams["apply_augmentation"]:
            # added the False for now, to avoid augmentation of any type
            wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens], dim=0)
        
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        
        # Feature extraction and normalization
        # wavs = self.modules.mean_var_norm_input(wavs, wav_lens)       

        # forward pass HF (possible: pre-trained) model
        # feats = self.modules.wav2vec2(wavs, wav_lens=wav_lens)
        feats = self.modules.wav2vec2(wavs)

        return feats, wav_lens

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings and output
        feats, lens = self.prepare_features(batch.sig, stage)

        # last dim will be used for pooling, 
        # StatisticsPooling uses 'lens'
        if hparams["avg_pool_class"] == "statpool":
            outputs = self.hparams.avg_pool(feats, lens)
        elif hparams["avg_pool_class"] == "avgpool":
            outputs = self.hparams.avg_pool(feats)
            # this uses a kernel, thus the output dim is not 1 (mean to reduce)
            outputs = outputs.mean(dim=1)
        else:
            outputs = self.hparams.avg_pool(feats)

        # preparing outputs
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.modules.preout_mlp(outputs)
        outputs_main = self.modules.output_mlp(outputs)
        outputs_main = self.hparams.log_softmax(outputs_main)
        outputs_adv = self.modules.output_mlp_adv(outputs[:self.hparams.batch_size, :])
        outputs_adv = self.hparams.log_softmax(outputs_adv)

        return outputs_main, outputs_adv, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions_acc, predictions_spk, lens = inputs

        # get the targets from the batch
        targets_acc = batch.accent_encoded.data
        # targets_spk = batch.speaker_encoded.data

        # to meet the input form of nll loss
        targets_acc = targets_acc.squeeze(1)
        # targets_spk = targets_spk.squeeze(1)
        uniform_log_prob_spk = math.log(1./self.hparams.n_speakers)
        # uniform_log_prob_spk = 1./self.hparams.n_speakers
        targets_spk = torch.ones(predictions_spk.shape, device=predictions_spk.device) * uniform_log_prob_spk

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hparams["apply_augmentation"]:
            targets_acc = torch.cat([targets_acc, targets_acc], dim=0)
            lens_acc = torch.cat([lens, lens], dim=0)

            # if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            #     self.hparams.lr_annealing.on_batch_end(self.optimizer)

        # get the final loss
        if hparams["label_smoothing"]:
            loss_main = self.hparams.compute_cost(predictions_acc, targets_acc, label_smoothing=hparams["label_smoothing"])
        else:
            loss_main = self.hparams.compute_cost(predictions_acc, targets_acc)
        loss_adv = self.hparams.compute_cost_adv(predictions_spk, targets_spk)
        loss = loss_main + self.hparams.weight_adv * loss_adv

        # append the metrics for evaluation
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions_acc, targets_acc)
            self.error_metrics2.append(batch.id, predictions_acc.argmax(-1), targets_acc)
            
            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(predictions_acc, targets_acc.view(1, -1), lens)
            self.acc_metric2.append(predictions_acc.argmax(-1), targets_acc.view(1, -1), lens)

            targets_spk = batch.speaker_encoded.data
            targets_spk = targets_spk.squeeze(1)
            self.error_metrics3.append(batch.id, predictions_spk, targets_spk)
            self.error_metrics4.append(batch.id, predictions_spk.argmax(-1), targets_spk)
            self.acc_metric3.append(predictions_spk, targets_spk.view(1, -1), lens)
            self.acc_metric4.append(predictions_spk.argmax(-1), targets_spk.view(1, -1), lens)

        return loss
    
    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""
        should_step = self.step % self.grad_accumulation_factor == 0

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        with self.no_sync(not should_step):
            (loss / self.grad_accumulation_factor).backward()
        if should_step:
            if self.check_gradients(loss):
                self.wav2vec2_optimizer.step()
                self.optimizer.step()
            self.wav2vec2_optimizer.zero_grad()
            self.optimizer.zero_grad()
            self.optimizer_step += 1

        self.on_fit_batch_end(batch, predictions, loss, should_step)
        return loss.detach().cpu()

    
    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()


    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            self.acc_metric = self.hparams.acc_computer()
            self.error_metrics2 = self.hparams.error_stats()
            self.acc_metric2 = self.hparams.acc_computer()
            self.error_metrics3 = self.hparams.error_stats()
            self.acc_metric3 = self.hparams.acc_computer()
            self.error_metrics4 = self.hparams.error_stats()
            self.acc_metric4 = self.hparams.acc_computer()


    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        
        stage_stats = {"loss": stage_loss}
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            # self.train_stats = stage_stats
            self.train_loss = stage_loss
        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["ACC_acc"] = self.acc_metric.summarize()
            stage_stats["error_rate_acc"] = self.error_metrics.summarize("average")
            stage_stats["ACC_spk"] = self.acc_metric3.summarize()
            stage_stats["error_rate_spk"] = self.error_metrics3.summarize("average")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # ipdb.set_trace()
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["error_rate_acc"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stage_stats["error_rate_acc"])
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            steps = self.optimizer_step

            # The train_logger writes a summary to stdout and to the logfile.
            epoch_stats = {
                "epoch": epoch,
                "lr": old_lr,
                "wave2vec_lr": old_lr_wav2vec2,
                "steps": steps,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats={"loss": self.train_loss},
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC_acc": stage_stats["ACC_acc"], "ACC_spk": stage_stats["ACC_spk"], "epoch": epoch},
                max_keys=["ACC_acc"],
                num_to_keep=1,
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec2_optimizer.zero_grad(set_to_none)
        self.optimizer.zero_grad(set_to_none)

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `common_accent_prepare` to have been called before this,
    so that the `train.csv`, `valid.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """
    
    # 1. Define train/valid/test datasets
    data_folder = hparams["data_folder"]
    csv_folder = hparams["csv_prepared_folder"]
    train_csv = os.path.join(csv_folder, "train" + ".csv")
    valid_csv = os.path.join(csv_folder, "dev_unseen" + ".csv")
    test_csv = os.path.join(csv_folder, "test_unseen" + ".csv")

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=train_csv, replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=valid_csv, replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")
    
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=test_csv, replacements={"data_root": data_folder},
    )
    # We also sort the test data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'accent01': 0, 'accent02': 1, ..)
    accent_encoder = sb.dataio.encoder.CategoricalEncoder()
    speaker_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        # info = torchaudio.info(wav)
        # sig = sb.dataio.dataio.read_audio(wav)
        # sig = torchaudio.transforms.Resample(
        #     info.sample_rate, hparams["sample_rate"],
        # )(sig)
        sig, _ = librosa.load(wav, sr=hparams["sample_rate"])
        sig = torch.tensor(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("accent", "speaker")
    @sb.utils.data_pipeline.provides("accent", "accent_encoded", "speaker", "speaker_encoded")
    def label_pipeline(accent, speaker):
        yield accent
        accent_encoded = accent_encoder.encode_label_torch(accent)
        yield accent_encoded
        yield speaker
        speaker_encoded = speaker_encoder.encode_label_torch(speaker)
        yield speaker_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "accent_encoded", "speaker_encoded"],
    )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    accent_encoder_file = os.path.join(hparams["save_folder"], "accent_encoder.txt")
    accent_encoder.load_or_create(
        path=accent_encoder_file,
        from_didatasets=[train_data],
        output_key="accent",
    )
    speaker_encoder_file = os.path.join(hparams["save_folder"], "speaker_encoder.txt")
    speaker_encoder.load_or_create(
        path=speaker_encoder_file,
        from_didatasets=[train_data],
        output_key="speaker",
    )
    speaker_encoder.add_unk()

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len_val"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )
    elif hparams["weighted_sampling"]:

        from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler

        weighted_sampler_hparams = hparams["weighted_sampler"]

        # generate weights for each sample - based on class imbalance for train & equal weights for valid
        train_sampler_weights = torch.ones(len(train_data))
        for i, key in enumerate(train_data.data_ids):
            accent = train_data.data[key]["accent"]
            train_sampler_weights[i] = hparams["weighted_sampler"]["accent_weights"][accent]
        
        valid_sampler_weights = torch.ones(len(valid_data))

        # sampler
        train_sampler = ReproducibleWeightedRandomSampler(
            weights=train_sampler_weights,
            num_samples=len(train_data),
            replacement=weighted_sampler_hparams["replacement"],
        )

        valid_sampler = ReproducibleWeightedRandomSampler(
            weights=valid_sampler_weights,
            num_samples=len(valid_data),
            replacement=False, #
        )

        # batch sampler
        from torch.utils.data import BatchSampler

        train_batch_sampler = BatchSampler(
            sampler=train_sampler,
            batch_size=hparams["train_dataloader_opts"]["batch_size"],
            drop_last=True
        )

        valid_batch_sampler = BatchSampler(
            sampler=valid_sampler,
            batch_size=hparams["valid_dataloader_opts"]["batch_size"],
            drop_last=False
        )

        print("Weighted sampler generated!!!")

    return (
        train_data,
        valid_data,
        test_data,
        train_batch_sampler,
        valid_batch_sampler,
        accent_encoder,
        speaker_encoder
    )

def get_pooling_layer(hparams):
    """function to get the pooling layer based on value in hparams file or CLI"""
    pooling = hparams["avg_pool_class"]
    
    # possible classes are statpool, adaptivepool, avgpool
    if pooling == "statpool":
        from speechbrain.nnet.pooling import StatisticsPooling
        pooling_layer = StatisticsPooling(return_std=False)
    elif pooling == "adaptivepool":
        from speechbrain.nnet.pooling import AdaptivePool
        pooling_layer = AdaptivePool(output_size=1)
    elif pooling == "avgpool":
        from speechbrain.nnet.pooling import Pooling1d
        pooling_layer = Pooling1d(pool_type="avg", kernel_size=3)
    else:
        raise ValueError("Pooling strategy must be in ['statpool', 'adaptivepool', 'avgpool']")
    hparams["avg_pool"] = pooling_layer

    return hparams

# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
        
    # defining the Pooling strategy based on hparams file:
    hparams = get_pooling_layer(hparams)

    # Create dataset objects "train", "valid", and "test", train/val samples and accent_encoder
    (
        train_data,
        valid_data,
        test_data,
        train_bsampler,
        valid_bsampler,
        accent_encoder,
        speaker_encoder
    ) = dataio_prep(hparams)

    # Load the Wav2Vec 2.0 model
    hparams["wav2vec2"] = hparams["wav2vec2"].to(run_opts["device"])
    # freeze the feature extractor part when unfreezing
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

    # Initialize the Brain object to prepare for mask training.
    aid_brain = AID(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    aid_brain.fit(
        aid_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Load the best checkpoint for evaluation
    test_stats = aid_brain.evaluate(
        test_set=test_data,
        min_key="error_rate",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

