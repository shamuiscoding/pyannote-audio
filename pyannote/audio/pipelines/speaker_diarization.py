# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Speaker diarization pipelines"""

import functools
import itertools
import math
import textwrap
import warnings
from typing import Callable, Mapping, Optional, Text, Union, Dict, Tuple

import numpy as np
import torch
from einops import rearrange
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import silhouette_score
from pyannote.core import Annotation, SlidingWindowFeature, Segment
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import ParamDict, Uniform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_model,
)
from pyannote.audio.pipelines.utils.diarization import set_num_speakers
from pyannote.audio.utils.signal import binarize


class ConfidenceCalculator:
    def __init__(self, embeddings, centroids):
        self.embeddings = embeddings
        self.centroids = centroids
        self.gmms = None

    def fit_gaussians(self, embeddings, cluster_labels):
        """Fit GMMs for each cluster for density-based confidence"""
        self.gmms = {}
        for i in range(np.max(cluster_labels) + 1):
            cluster_embeddings = embeddings[cluster_labels == i]
            if len(cluster_embeddings) > 0:
                gmm = GaussianMixture(n_components=1, covariance_type="full")
                gmm.fit(cluster_embeddings.reshape(-1, embeddings.shape[-1]))
                self.gmms[i] = gmm

    def compute_confidence(
        self, diarization, segmentations, embeddings, centroids, label_to_idx
    ) -> Dict[Segment, Dict[str, float]]:
        """Calculate confidence scores for each segment and speaker"""
        confidence_scores = {}
        print("segmentations", segmentations)
        print("embeddings", embeddings)
        print("centroids", centroids)
        print("label_to_idx", label_to_idx)

        # Process each segment in the diarization
        for segment, track, label in diarization.itertracks(yield_label=True):
            # Find embeddings that fall within this segment
            print("segment", segment, "track", track, "label", label)
            segment_start_idx = int(segment.start / segmentations.sliding_window.step)
            segment_end_idx = int(segment.end / segmentations.sliding_window.step)

            # Get relevant embeddings and probabilities
            segment_embeddings = embeddings[segment_start_idx:segment_end_idx]
            segment_probs = segmentations.data[segment_start_idx:segment_end_idx]

            if len(segment_embeddings) == 0:
                continue

            # Calculate mean probability for this segment
            speaker_idx = label_to_idx[label]
            print("speaker_idx", speaker_idx)
            mean_prob = np.mean(segment_probs[:, speaker_idx])
            print("mean_prob", mean_prob)

            # Store the confidence score
            confidence_scores[segment] = {
                label: float(mean_prob)  # Convert to native Python float
            }

        return confidence_scores


def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class SpeakerDiarization(SpeakerDiarizationMixin, Pipeline):
    """Speaker diarization pipeline with confidence calculation."""

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation@2022.07",
        segmentation_step: float = 0.1,
        embedding: PipelineModel = "speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e",
        embedding_exclude_overlap: bool = False,
        clustering: str = "AgglomerativeClustering",
        embedding_batch_size: int = 1,
        segmentation_batch_size: int = 1,
        der_variant: Optional[dict] = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()

        self.segmentation_model = segmentation
        model: Model = get_model(segmentation, use_auth_token=use_auth_token)

        self.segmentation_step = segmentation_step

        self.embedding = embedding
        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap

        self.klustering = clustering

        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}

        segmentation_duration = model.specifications.duration
        self._segmentation = Inference(
            model,
            duration=segmentation_duration,
            step=self.segmentation_step * segmentation_duration,
            skip_aggregation=True,
            batch_size=segmentation_batch_size,
        )

        if self._segmentation.model.specifications.powerset:
            self.segmentation = ParamDict(
                min_duration_off=Uniform(0.0, 1.0),
            )

        else:
            self.segmentation = ParamDict(
                threshold=Uniform(0.1, 0.9),
                min_duration_off=Uniform(0.0, 1.0),
            )

        if self.klustering == "OracleClustering":
            metric = "not_applicable"
        else:
            self._embedding = PretrainedSpeakerEmbedding(
                self.embedding, use_auth_token=use_auth_token
            )
            self._audio = Audio(sample_rate=self._embedding.sample_rate, mono="downmix")
            metric = self._embedding.metric

        try:
            Klustering = Clustering[clustering]
        except KeyError:
            raise ValueError(
                f'clustering must be one of [{", ".join(list(Clustering.__members__))}]'
            )
        self.clustering = Klustering.value(metric=metric)

        self._expects_num_speakers = self.clustering.expects_num_clusters

    @property
    def segmentation_batch_size(self) -> int:
        return self._segmentation.batch_size

    @segmentation_batch_size.setter
    def segmentation_batch_size(self, batch_size: int):
        self._segmentation.batch_size = batch_size

    def default_parameters(self):
        raise NotImplementedError()

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1

    @property
    def CACHED_SEGMENTATION(self):
        return "training_cache/segmentation"

    def get_segmentations(self, file, hook=None) -> SlidingWindowFeature:
        """Apply segmentation model"""
        if hook is not None:
            hook = functools.partial(hook, "segmentation", None)

        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file, hook=hook)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file, hook=hook)

        return segmentations

    def get_embeddings(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
        hook: Optional[Callable] = None,
    ):
        """Extract embeddings for each (chunk, speaker) pair."""
        if self.training:
            cache = file.get("training_cache/embeddings", dict())
            if ("embeddings" in cache) and (
                self._segmentation.model.specifications.powerset
                or (cache.get("segmentation.threshold") == self.segmentation.threshold)
            ):
                return cache["embeddings"]

        duration = binary_segmentations.sliding_window.duration
        num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

        if exclude_overlap:
            min_num_samples = self._embedding.min_num_samples
            num_samples = duration * self._embedding.sample_rate
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

            clean_frames = 1.0 * (
                np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
            )
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data * clean_frames,
                binary_segmentations.sliding_window,
            )
        else:
            min_num_frames = -1
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data, binary_segmentations.sliding_window
            )

        def iter_waveform_and_mask():
            for (chunk, masks), (_, clean_masks) in zip(
                binary_segmentations, clean_segmentations
            ):
                waveform, _ = self._audio.crop(
                    file,
                    chunk,
                    duration=duration,
                    mode="pad",
                )

                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

                for mask, clean_mask in zip(masks.T, clean_masks.T):
                    if np.sum(clean_mask) > min_num_frames:
                        used_mask = clean_mask
                    else:
                        used_mask = mask

                    yield waveform[None], torch.from_numpy(used_mask)[None]

        batches = batchify(
            iter_waveform_and_mask(),
            batch_size=self.embedding_batch_size,
            fillvalue=(None, None),
        )

        batch_count = math.ceil(num_chunks * num_speakers / self.embedding_batch_size)
        embedding_batches = []

        if hook is not None:
            hook("embeddings", None, total=batch_count, completed=0)

        for i, batch in enumerate(batches, 1):
            waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))
            waveform_batch = torch.vstack(waveforms)
            mask_batch = torch.vstack(masks)

            embedding_batch: np.ndarray = self._embedding(
                waveform_batch, masks=mask_batch
            )
            embedding_batches.append(embedding_batch)

            if hook is not None:
                hook("embeddings", embedding_batch, total=batch_count, completed=i)

        embedding_batches = np.vstack(embedding_batches)

        embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)
        if self.training:
            if self._segmentation.model.specifications.powerset:
                file["training_cache/embeddings"] = {
                    "embeddings": embeddings,
                }
            else:
                file["training_cache/embeddings"] = {
                    "segmentation.threshold": self.segmentation.threshold,
                    "embeddings": embeddings,
                }

        return embeddings

    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        hard_clusters: np.ndarray,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        """Build final discrete diarization out of clustered segmentation."""
        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        num_clusters = np.max(hard_clusters) + 1
        clustered_segmentations = np.nan * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )

        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(hard_clusters, segmentations)
        ):
            for k in np.unique(cluster):
                if k == -2:
                    continue
                clustered_segmentations[c, :, k] = np.max(
                    segmentation[:, cluster == k], axis=1
                )

        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )

        return self.to_diarization(clustered_segmentations, count)

    def apply(
        self,
        file: AudioFile,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
        return_confidence: bool = True,
        hook: Optional[Callable] = None,
    ):
        hook = self.setup_hook(file, hook=hook)

        num_speakers, min_speakers, max_speakers = set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        if self._expects_num_speakers and num_speakers is None:
            if isinstance(file, Mapping) and "annotation" in file:
                num_speakers = len(file["annotation"].labels())
            else:
                raise ValueError(
                    f"num_speakers must be provided when using {self.klustering} clustering"
                )

        segmentations = self.get_segmentations(file, hook=hook)
        hook("segmentation", segmentations)
        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        # binarize segmentation
        if self._segmentation.model.specifications.powerset:
            binarized_segmentations = segmentations
        else:
            binarized_segmentations: SlidingWindowFeature = binarize(
                segmentations,
                onset=self.segmentation.threshold,
                initial_state=False,
            )

        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation.model.receptive_field,
            warm_up=(0.0, 0.0),
        )
        hook("speaker_counting", count)

        if np.nanmax(count.data) == 0.0:
            diarization = Annotation(uri=file["uri"])
            if return_embeddings:
                return diarization, np.zeros((0, self._embedding.dimension))
            return diarization

        if not return_embeddings and max_speakers < 2:
            hard_clusters = np.zeros((num_chunks, local_num_speakers), dtype=np.int8)
            embeddings = None
            centroids = None
        else:
            if self.klustering == "OracleClustering" and not return_embeddings:
                embeddings = None
            else:
                embeddings = self.get_embeddings(
                    file,
                    binarized_segmentations,
                    exclude_overlap=self.embedding_exclude_overlap,
                    hook=hook,
                )
                hook("embeddings", embeddings)

            hard_clusters, _, centroids = self.clustering(
                embeddings=embeddings,
                segmentations=binarized_segmentations,
                num_clusters=num_speakers,
                min_clusters=min_speakers,
                max_clusters=max_speakers,
                file=file,
                frames=self._segmentation.model.receptive_field,
            )

        num_different_speakers = np.max(hard_clusters) + 1
        if (
            num_different_speakers < min_speakers
            or num_different_speakers > max_speakers
        ):
            warnings.warn(
                textwrap.dedent(
                    f"""
                The detected number of speakers ({num_different_speakers}) is outside
                the given bounds [{min_speakers}, {max_speakers}]. This can happen if the
                given audio file is too short to contain {min_speakers} or more speakers.
                Try to lower the desired minimal number of speakers.
                """
                )
            )

        count.data = np.minimum(count.data, max_speakers).astype(np.int8)

        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        hard_clusters[inactive_speakers] = -2
        discrete_diarization = self.reconstruct(segmentations, hard_clusters, count)
        hook("discrete_diarization", discrete_diarization)

        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=self.segmentation.min_duration_off,
        )
        diarization.uri = file["uri"]

        if "annotation" in file and file["annotation"]:
            _, mapping = self.optimal_mapping(
                file["annotation"], diarization, return_mapping=True
            )
            mapping = {key: mapping.get(key, key) for key in diarization.labels()}
        else:
            mapping = {
                label: expected_label
                for label, expected_label in zip(diarization.labels(), self.classes())
            }

        diarization = diarization.rename_labels(mapping=mapping)

        if not return_embeddings and not return_confidence:
            return diarization

        if centroids is None:
            # OracleClustering with no embeddings
            if return_confidence:
                # No confidence available
                return diarization, {}
            return diarization, None

        # Adjust centroids if fewer centroids than speakers
        if len(diarization.labels()) > centroids.shape[0]:
            centroids = np.pad(
                centroids, ((0, len(diarization.labels()) - centroids.shape[0]), (0, 0))
            )

        # Re-order centroids
        inverse_mapping = {label: index for index, label in mapping.items()}
        centroids = centroids[
            [inverse_mapping[label] for label in diarization.labels()]
        ]

        # Build label_to_idx
        label_to_idx = {label: i for i, label in enumerate(diarization.labels())}

        breakpoint()

        if return_confidence:
            calc = ConfidenceCalculator(embeddings, centroids)
            # Optionally, if you want GMM-based density confidence:
            calc.fit_gaussians(embeddings, hard_clusters)

            confidence = calc.compute_confidence(
                diarization=diarization,
                segmentations=segmentations,
                embeddings=embeddings,
                centroids=centroids,
                label_to_idx=label_to_idx,
            )

            if return_embeddings:
                return diarization, centroids, confidence
            return diarization, confidence
        else:
            if return_embeddings:
                return diarization, centroids
            return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(**self.der_variant)
