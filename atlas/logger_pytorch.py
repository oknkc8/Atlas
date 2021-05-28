# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

import os

from tensorboardX import SummaryWriter
import torch


class MeshWriter:
    """ Saves mesh to logdir during training"""

    def __init__(self, save_path):
        self._save_path = os.path.join(save_path, "train_viz")
        os.makedirs(self._save_path, exist_ok=True)
        os.makedirs(os.path.join(self._save_path, 'valid'), exist_ok=True)
  
    def save_mesh(self, tsdf, name):
        if 'semseg' in tsdf.attribute_vols:
            mesh = tsdf.get_mesh('semseg')
        else:
            mesh = tsdf.get_mesh()

        # TODO: include epoch # and train/val
        mesh.export(os.path.join(self._save_path, name))


class AtlasLogger():
    def __init__(self, save_path):
        super().__init__()

        self.mesh_writer = MeshWriter(save_path)
        self.loss_writer = SummaryWriter(save_path)