#!/usr/bin/env python3
import logging
import os
import sys

import librosa
import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from tqdm import tqdm
import fsspec
from collections import Counter
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt

"""Recipe for performing inference on Accent Classification system with CommonVoice Accent.

To run this recipe, do the following:
> python extract_embeddings_GenAID.py extract_embeddings_GenAID_v6.yaml

Author
------
 * Created by Juan Pablo Zuluaga 2023
 * Modified by Jinzuomu Zhong 2024
"""

logger = logging.getLogger(__name__)


# Brain class for Accent ID training
class AccID_inf(sb.Brain):
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
        # if stage == sb.Stage.TRAIN and hparams["apply_augmentation"]:
        #     # added the False for now, to avoid augmentation of any type
        #     wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
        #     wavs = torch.cat([wavs, wavs_noise], dim=0)
        #     wav_lens = torch.cat([wav_lens, wav_lens], dim=0)
        
        #     if hasattr(self.hparams, "augmentation"):
        #         wavs = self.hparams.augmentation(wavs, wav_lens)
        
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

        # embedding
        outputs = outputs.view(outputs.shape[0], -1)
        try:
            outputs_2 = self.modules.preout_mlp(outputs)
        except:
            outputs_2 = outputs
        # prediction
        outputs_3 = self.modules.output_mlp(outputs_2)
        probs = self.hparams.log_softmax(outputs_3)
        classes = torch.argmax(probs, dim=1)

        return outputs_2, lens, classes, probs

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

        lens, predictions = inputs

        # get the targets from the batch
        targets = batch.accent_encoded.data

        # to meet the input form of nll loss
        targets = targets.squeeze(1)

        # get the final loss
        loss = self.hparams.compute_cost(predictions, targets)

        # append the metrics for evaluation
        if stage != sb.Stage.TRAIN:
            # ipdb.set_trace()
            self.error_metrics.append(batch.id, predictions, targets)
            self.error_metrics2.append(batch.id, predictions.argmax(-1), targets)

        return loss

    def evaluate_batch(self, batch, stage, mode="inf"):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            embeddings, lens, classes, predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives((lens, predictions), batch, stage=stage)
        if mode == "extract":
            return embeddings.detach(), classes.detach()
        else:
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
            self.error_metrics2 = self.hparams.error_stats2()
    
    def extract(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        accent_mapping = {}
        class_count = Counter()
        accent_predictions = {}
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                embeddings, classes = self.evaluate_batch(batch, stage=sb.Stage.TEST, mode="extract")
                class_count.update(classes.tolist())
                for utt_id, speaker, embedding, accent in zip(batch.utt_id, batch.speaker, embeddings, classes):
                    accent_mapping[utt_id] = {}
                    # accent_mapping[utt_id]["name"] = "LTTS_" + speaker
                    # accent_mapping[utt_id]["name"] = "VCTK_" + speaker
                    accent_mapping[utt_id]["name"] = speaker
                    accent_mapping[utt_id]["embedding"] = embedding.tolist()
                    accent_predictions[utt_id] = accent
                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)
        self.step = 0
        return class_count, accent_predictions, accent_mapping

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `common_accent_prepare` to have been called before this,
    so that the `train.csv`, `dev.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "dev" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        # sig, _ = torchaudio.load(wav)
        # sig = sig.transpose(0, 1).squeeze(1)
        # Problem with Torchaudio while reading MP3 files (CommonVoice)
        sig, _ = librosa.load(wav, sr=hparams["sample_rate"])
        sig = torch.tensor(sig)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("accent")
    @sb.utils.data_pipeline.provides("accent", "accent_encoded")
    def label_pipeline(accent):
        yield accent
        accent_encoded = accent_encoder.encode_label_torch(accent)
        yield accent_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["all_file_paths"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=os.path.join(hparams["csv_prepared_folder"], dataset + ".csv"),
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "utt_id", "speaker", "sig", "accent_encoded"],
        )
        # filtering out recordings with more than max_audio_length allowed
        datasets[dataset] = datasets[dataset].filtered_sorted(
            key_max_value={"duration": hparams["max_audio_length"]},
        )

    return datasets

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

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create output directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    hparams = get_pooling_layer(hparams)
    
    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'accent01': 0, 'accent02': 1, ..)
    accent_encoder = sb.dataio.encoder.CategoricalEncoder()
    # Load label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    accent_encoder_file = os.path.join(hparams["pretrainer"].paths["label_encoder"])
    accent_encoder.load_or_create(
        path=accent_encoder_file,
        output_key="accent",
    )
    # accent_encoder.add_unk()

    # Create dataset objects "train", "dev", and "test" and accent_encoder
    datasets = dataio_prep(hparams)

    # Fetch and laod pretrained modules
    sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()

    # Initialize the Brain object to prepare for performing infernence.
    accid_brain = AccID_inf(
        modules=hparams["modules"],
        hparams=hparams,
    )

    # Function that actually prints the output. you can modify this to get some other information
    
    for dataset in ["all_file_paths"]:
        test_stats = accid_brain.evaluate(
            test_set=datasets[dataset],
            min_key="error",
            test_loader_kwargs=hparams["test_dataloader_options"],
        )

        class_count, accent_predictions, accent_mapping = accid_brain.extract(
            test_set=datasets[dataset],
            test_loader_kwargs=hparams["test_dataloader_options"],
        )
        print(class_count)

        with open(os.path.join(hparams["output_folder"], "accent_predictions.txt"), "w") as f:
            for utt_id, accent in accent_predictions.items():
                accent = accent_encoder.ind2lab[accent.cpu().item()]
                f.write(utt_id+"\t"+accent+"\r\n")
        with fsspec.open(os.path.join(hparams["output_folder"], "accents.pth"), "wb") as f:
            torch.save(accent_mapping, f)
