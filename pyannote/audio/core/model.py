# MIT License
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from importlib import import_module
from typing import Any, Dict, Optional, Text

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pyannote.audio import __version__
from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import Problem, Task
from pytorch_lightning.utilities.cloud_io import load as pl_load
from semver import VersionInfo


class Model(pl.LightningModule):
    """Base model

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    task : Task, optional
        Task addressed by the model. Only provided when training the model.
        A model should be `load_from_checkpoint`-able without a task as
        `on_load_checkpoint` hook takes care of calling `setup`.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__()

        # set-up audio IO
        assert (
            num_channels == 1
        ), "Only mono audio is supported for now (num_channels = 1)"
        self.hparams.sample_rate = sample_rate
        self.hparams.num_channels = num_channels
        self.audio = Audio(sample_rate=self.hparams.sample_rate, mono=True)

        # set task attribute when available (i.e. at training time)
        # and also tell the task what kind of audio is expected from
        # the model
        if task is not None:
            self.task = task
            self.task.audio = self.audio

    def build(self):
        # use this method to add task-dependent layers to the model
        # (e.g. the final classification and activation layers)
        pass

    def setup(self, stage=None):

        if stage == "fit":

            # TODO: move these to self.hparams.task

            # keep track of the classes here because it is used
            # to setup the final classification layer (even when stage != fit)
            self.hparams.classes = self.task.specifications.classes

            # keep track of the type of problem here because it is used
            # to setup the final activation layer (even when stage != fit)
            self.hparams.problem = self.task.specifications.problem

            # any other common parameters should be saved?
            # maybe the class of the model (and pyannote.audio semantic version?)
            # so that it can be loaded without knowing what type of model it is.
            # this would probably make distributing pretrained models much easier.

        else:
            # should we do something specific when stage != fit?
            # hparams.classes and hparams.problem should already exist
            # because they should have been loaded on_load_checkpoint
            pass

        # add task-dependent layers to the model
        # (e.g. the final classification and activation layers)
        self.build()

        if stage == "fit":

            # let task know about the shape of model output
            # so that its dataloader knows how to generate targets
            self.task.example_output_array = self.forward(
                self.task.example_input_array()
            )

    @staticmethod
    def check_version(library: Text, theirs: Text, mine: Text):
        theirs = VersionInfo.parse(theirs)
        mine = VersionInfo.parse(mine)
        if theirs.major != mine.major:
            warnings.warn(
                f"Model was trained with {library} {theirs}, yours is {mine}. "
                f"Bad things will probably happen unless you update {library} to {theirs.major}.x."
            )
        if theirs.minor > mine.minor:
            warnings.warn(
                f"Model was trained with {library} {theirs}, yours is {mine}. "
                f"This should be OK but you might want to update {library}."
            )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):

        try:
            self.check_version(
                "pyannote.audio",
                checkpoint["pyannote.audio"]["versions"]["pyannote.audio"],
                __version__,
            )
        except ValueError:
            warnings.warn("FIXME: could not check pyannote.audio version")

        self.check_version(
            "torch",
            checkpoint["pyannote.audio"]["versions"]["torch"],
            torch.__version__,
        )
        self.check_version(
            "pytorch-lightning", checkpoint["pytorch-lightning_version"], pl.__version__
        )

        # only hyper-parameters defined in __init__ are loaded automatically.
        # therefore, we have to manually load hyper-parameters that were
        # defined during setup()
        self.hparams.classes = checkpoint["hyper_parameters"]["classes"]
        self.hparams.problem = checkpoint["hyper_parameters"]["problem"]
        # TODO: would have to check pytorch-lightning documentation to see
        # if we can get rid of this... it is weird that only "some" parameters
        # in self.hparams are assigned at __init__ time...

        # now that setup()-defined hyper-parameters are available,
        # we can actually setup() the model.
        self.setup()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):

        #  put everything pyannote.audio-specific under pyannote.audio
        #  to avoid any future conflicts with pytorch-lightning updates
        checkpoint["pyannote.audio"] = {
            "versions": {
                "torch": torch.__version__,
                "pyannote.audio": __version__,
            },
            "model": {
                "module": self.__class__.__module__,
                "class": self.__class__.__name__,
            },
        }

        # TODO give self.task a chance to save some hyper-parameters as well (e.g. chunk duration)
        # TODO self.task.on_save_checkpoint(checkpoint)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        msg = "Class {self.__class__.__name__} should define a `forward` method."
        raise NotImplementedError(msg)

    # convenience function to automate the choice of the final activation function
    def default_activation(self) -> nn.Module:

        if self.hparams.problem == Problem.MONO_LABEL_CLASSIFICATION:
            return nn.LogSoftmax(dim=-1)

        elif self.hparams.problem == Problem.MULTI_LABEL_CLASSIFICATION:
            return nn.Sigmoid()

        else:
            msg = "TODO: implement default activation for other types of problems"
            raise NotImplementedError(msg)

    # training step logic is defined by the task because the
    # model does not really need to know how it is being used.
    def training_step(self, batch, batch_idx):
        return self.task.training_step(self, batch, batch_idx)

    # optimizer is defined by the task for the same reason as above
    def configure_optimizers(self):
        return self.task.configure_optimizers(self)


def load_from_checkpoint(checkpoint_path: str, map_location=None) -> Model:
    """Load model from checkpoint

    Parameters
    ----------
    checkpoint_path: str
        Path to checkpoint. This can also be a URL.

    Returns
    -------
    model : Model
        Model
    """

    # obtain model class from the checkpoint
    checkpoint = pl_load(checkpoint_path, map_location=map_location)

    module_name: str = checkpoint["pyannote.audio"]["model"]["module"]
    module = import_module(module_name)

    class_name: str = checkpoint["pyannote.audio"]["model"]["class"]
    Klass: Model = getattr(module, class_name)

    return Klass.load_from_checkpoint(checkpoint_path)
