#!/usr/bin/env python3
import logging
import os
import sys

import librosa
import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt

"""Recipe for performing inference on an Accent Identification (AID) system GenAID, with CommonAccent dataset.

To run this recipe, do the following:
> python inference_GenAID.py inference_GenAID_v6.yaml

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

        # preparing outputs
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.modules.preout_mlp(outputs)
        outputs = self.modules.output_mlp(outputs)
        outputs = self.hparams.log_softmax(outputs)

        return outputs, lens

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

        predictions, lens = inputs

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
            self.error_metrics2 = self.hparams.error_stats2()

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
    for dataset in ["dev_unseen", "test_unseen", "dev_seen", "test_seen"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=os.path.join(hparams["csv_prepared_folder"], dataset + ".csv"),
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "accent_encoded"],
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
    def print_confusion_matrix(AccID_object, set_name="dev"):
        """pass the object what contains the stats"""

        # get the scores after running the forward pass
        y_true_val = torch.cat([_.unsqueeze(0) for _ in AccID_object.error_metrics2.labels]).tolist()
        y_pred_val = torch.cat([_.unsqueeze(0) for _ in AccID_object.error_metrics2.scores]).tolist()

        # get the values of the items from the dictionary
        y_true = [accent_encoder.ind2lab[i] for i in y_true_val]
        y_pred = [accent_encoder.ind2lab[i] for i in y_pred_val]
        # retrieve a list of classes
        classes = [i[1] for i in accent_encoder.ind2lab.items()]

        with open(
            f"{hparams['output_folder']}/classification_report_{set_name}.txt", "w"
        ) as f:
            f.write(classification_report(y_true, y_pred))

        # create the confusion matrix and plot it
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        # modification of the class labels for nicer plot
        classes_display = [_[0].upper()+_[1:] for _ in classes]
        classes_display[0] = "American"
        classes_display[5] = "South African"
        classes_display[10] = "Hong Kong"
        classes_display[12] = "New Zealand"

        plt.rcParams.update({'font.size': 16})

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_display)

        fig, ax = plt.subplots(figsize=(10,10))

        # Deactivate default colorbar
        disp.plot(ax=ax, colorbar=False)
        disp.ax_.tick_params(axis="x", labelrotation=75)
        
        plt.tight_layout(rect=(0,0,0.9,1))
        
        # Adding custom colorbar
        cax = fig.add_axes([ax.get_position().x1+0.02,ax.get_position().y0,0.03,ax.get_position().height])
        plt.colorbar(disp.im_, cax=cax)
        plt.savefig(
            f"{hparams['output_folder']}/conf_mat_{set_name}.png", dpi=1000
        )
    
    for dataset in ["dev_unseen", "test_unseen", "dev_seen", "test_seen"]:
        test_stats = accid_brain.evaluate(
            test_set=datasets[dataset],
            min_key="error",
            test_loader_kwargs=hparams["test_dataloader_options"],
        )
        print_confusion_matrix(accid_brain, set_name=dataset)
