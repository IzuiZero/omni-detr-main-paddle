# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import paddle

def to_cuda(samples, targets, device):
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return samples, targets


def to_cuda_semi(samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes, device):
    samples_q = samples_q.to(device)
    samples_k = samples_k.to(device)
    targets_q = [{k: v.to(device) for k, v in t.items()} for t in targets_q]
    targets_k = [{k: v.to(device) for k, v in t.items()} for t in targets_k]
    return samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = paddle.device.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        with paddle.device.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)

    def next(self):
        if self.prefetch:
            paddle.device.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(paddle.device.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(paddle.device.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets



class data_prefetcher_semi():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = paddle.device.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples_q, self.next_targets_q, self.next_records_q, self.next_samples_k, self.next_targets_k, self.next_records_k, self.next_indicators, self.next_labeltypes = next(self.loader)
        except StopIteration:
            self.next_samples_q = None
            self.next_targets_q = None
            self.next_samples_k = None
            self.next_targets_k = None
            self.next_records_q = None
            self.next_records_k = None
            self.next_indicators = None
            self.next_labeltypes = None
            return
        with paddle.device.cuda.stream(self.stream):
            self.next_samples_q, self.next_targets_q, self.next_records_q, self.next_samples_k, self.next_targets_k, self.next_records_k, self.next_indicators, self.next_labeltypes = to_cuda_semi(self.next_samples_q, self.next_targets_q, self.next_records_q, self.next_samples_k, self.next_targets_k, self.next_records_k, self.next_indicators, self.next_labeltypes, self.device)

    def next(self):
        if self.prefetch:
            paddle.device.cuda.current_stream().wait_stream(self.stream)
            samples_q = self.next_samples_q
            targets_q = self.next_targets_q
            records_q = self.next_records_q
            samples_k = self.next_samples_k
            targets_k = self.next_targets_k
            records_k = self.next_records_k
            indicators = self.next_indicators
            labeltypes = self.next_labeltypes
            if samples_q is not None:
                samples_q.record_stream(paddle.device.cuda.current_stream())
            if samples_k is not None:
                samples_k.record_stream(paddle.device.cuda.current_stream())
            if targets_q is not None:
                for t in targets_q:
                    for k, v in t.items():
                        v.record_stream(paddle.device.cuda.current_stream())
            if targets_k is not None:
                for t in targets_k:
                    for k, v in t.items():
                        v.record_stream(paddle.device.cuda.current_stream())
            self.preload()
        else:
            try:
                samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes = next(self.loader)
                samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes = to_cuda_semi(samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes, self.device)
            except StopIteration:
                samples_q = None
                targets_q = None
                samples_k = None
                targets_k = None
                indicators = None
                labeltypes = None
        return samples_q, targets_q, records_q, samples_k, targets_k, records_k, indicators, labeltypes
