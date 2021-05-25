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

import itertools
import os

import torch
from torch import nn
from torch.nn import functional as F

from atlas.config import CfgNode
from atlas.data import ScenesDataset, collate_fn, parse_splits_list
from atlas.heads2d import PixelHeads
from atlas.heads3d import VoxelHeads
from atlas.backbone2d import build_backbone2d
from atlas.backbone3d import build_backbone3d
import atlas.transforms as transforms
from atlas.tsdf import coordinates, TSDF


def backproject(voxel_dim, voxel_size, origin, projection, features):
    """ Takes 2d features and fills them along rays in a 3d volume

    This function implements eqs. 1,2 in https://arxiv.org/pdf/2003.10432.pdf
    Each pixel in a feature image corresponds to a ray in 3d.
    We fill all the voxels along the ray with that pixel's features.

    Args:
        voxel_dim: size of voxel volume to construct (nx,ny,nz)
        voxel_size: metric size of each voxel (ex: .04m)
        origin: origin of the voxel volume (xyz position of voxel (0,0,0))
        projection: bx4x3 projection matrices (intrinsics@extrinsics)
        features: bxcxhxw  2d feature tensor to be backprojected into 3d

    Returns:
        volume: b x c x nx x ny x nz 3d feature volume
        valid:  b x 1 x nx x ny x nz volume.
                Each voxel contains a 1 if it projects to a pixel
                and 0 otherwise (not in view frustrum of the camera)
    """

    batch = features.size(0)
    channels = features.size(1)
    device = features.device
    nx, ny, nz = voxel_dim

    coords = coordinates(voxel_dim, device).unsqueeze(0).expand(batch,-1,-1) # bx3xhwd
    world = coords.type_as(projection) * voxel_size + origin.to(device).unsqueeze(2)
    world = torch.cat((world, torch.ones_like(world[:,:1]) ), dim=1) # homogeneous
    
    camera = torch.bmm(projection, world)
    px = (camera[:,0,:]/camera[:,2,:]).round().type(torch.long)
    py = (camera[:,1,:]/camera[:,2,:]).round().type(torch.long)
    pz = camera[:,2,:]

    # voxels in view frustrum
    height, width = features.size()[2:]
    valid = (px >= 0) & (py >= 0) & (px < width) & (py < height) & (pz>0) # bxhwd

    # put features in volume
    volume = torch.zeros(batch, channels, nx*ny*nz, dtype=features.dtype, 
                         device=device)
    for b in range(batch):
        volume[b,:,valid[b]] = features[b,:,py[b,valid[b]], px[b,valid[b]]]

    volume = volume.view(batch, channels, nx, ny, nz)
    valid = valid.view(batch, 1, nx, ny, nz)

    return volume, valid


class VoxelNet(nn.Module):
    """ Network architecture implementing ATLAS (https://arxiv.org/pdf/2003.10432.pdf)"""

    def __init__(self, hparams, device="cuda"):
        super().__init__()

        self.device = device

        # see config.py for details
        self.hparams = hparams

        # pytorch lightning does not support saving YACS CfgNone     
        self.cfg = CfgNode(self.hparams)
        cfg = self.cfg

        # networks
        self.backbone2d, self.backbone2d_stride = build_backbone2d(cfg)
        self.backbone2d = self.backbone2d.to(self.device)
        self.backbone3d = build_backbone3d(cfg)
        self.heads2d = PixelHeads(cfg, self.backbone2d_stride)
        self.heads3d = VoxelHeads(cfg)

        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)

        self.voxel_size = cfg.VOXEL_SIZE
        self.voxel_dim_train = cfg.VOXEL_DIM_TRAIN
        self.voxel_dim_val = cfg.VOXEL_DIM_VAL
        self.voxel_dim_test = cfg.VOXEL_DIM_TEST
        self.origin = torch.tensor([0,0,0]).view(1,3)

        self.batch_size_train = cfg.DATA.BATCH_SIZE_TRAIN
        self.num_frames_train = cfg.DATA.NUM_FRAMES_TRAIN
        self.num_frames_val = cfg.DATA.NUM_FRAMES_VAL
        self.frame_types = cfg.MODEL.HEADS2D.HEADS
        self.frame_selection = cfg.DATA.FRAME_SELECTION
        self.batch_backbone2d_time = cfg.TRAINER.BATCH_BACKBONE2D_TIME
        self.finetune3d = cfg.TRAINER.FINETUNE3D
        self.voxel_types = cfg.MODEL.HEADS3D.HEADS
        self.voxel_sizes = [int(cfg.VOXEL_SIZE*100)*2**i for i in 
                            range(len(cfg.MODEL.BACKBONE3D.LAYERS_DOWN)-1)]

        self.initialize_volume()


    def initialize_volume(self):
        """ Reset the accumulators.
        
        self.volume is a voxel volume containg the accumulated features
        self.valid is a voxel volume containg the number of times a voxel has
            been seen by a camera view frustrum
        """

        self.volume = 0
        self.valid = 0

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def feature_accumulation(self, projection, image=None, feature=None):
        """ Backprojects image features into 3D and accumulates them.

        This is the first half of the network which is run on every frame.
        Only pass one of image or feature. If image is passed 2D features
        are extracted from the image using self.backbone2d. When features
        are extracted external to this function pass features (used when 
        passing multiple frames through the backbone2d simultaniously
        to share BatchNorm stats).

        Args:
            projection: bx3x4 projection matrix
            image: bx3xhxw RGB image
            feature: bxcxh'xw' feature map (h'=h/stride, w'=w/stride)

        Feature volume is accumulated into self.volume and self.valid
        """

        assert ((image is not None and feature is None) or 
                (image is None and feature is not None))

        if feature is None:
            image = self.normalizer(image)
            feature = self.backbone2d(image)

        # backbone2d reduces the size of the images so we 
        # change intrinsics to reflect this
        projection = projection.clone()
        projection[:,:2,:] = projection[:,:2,:] / self.backbone2d_stride

        if self.training:
            voxel_dim = self.voxel_dim_train
        else:
            voxel_dim = self.voxel_dim_val
        volume, valid = backproject(voxel_dim, self.voxel_size, self.origin,
                                    projection, feature)

        if self.finetune3d:
            volume.detach_()
            valid.detach_()

        self.volume = self.volume + volume
        self.valid = self.valid + valid

    def refine_3d_cnn(self, targets=None):
        """ Refines accumulated features and regresses output TSDF.

        This is the second half of the network. It should be run once after
        all frames have been accumulated. It may also be run more fequently
        to visualize incremental progress.

        Args:
            targets: used to compare network output to ground truth

        Returns:
            tuple of dicts ({outputs}, {losses})
                if targets is None, losses is empty
        """

        volume = self.volume/self.valid

        # remove nans (where self.valid==0)
        volume = volume.transpose(0,1)
        volume[:,self.valid.squeeze(1)==0]=0
        volume = volume.transpose(0,1)

        x = self.backbone3d(volume)
        return self.heads3d(x, targets)


    def forward(self, batch):
        """ Wraps feature_accumulation() and refine_3d_cnn() into a single call.

        Args:
            batch: a dict from the dataloader

        Returns:
            see self.refine_3d_cnn
        """

        self.initialize_volume()

        image = batch['image']
        projection = batch['projection'].to(self.device)

        # get targets if they are in the batch
        targets3d = {key:value.to(self.device) for key, value in batch.items() if key[:3]=='vol'}
        targets3d = targets3d if targets3d else None
        
        # transpose batch and time so we can accumulate sequentially
        images = image.transpose(0,1)
        projections = projection.transpose(0,1)

        if (not self.batch_backbone2d_time) or (not self.training) or self.finetune3d:
            # run images through 2d cnn sequentially and backproject and accumulate
            # no use batchnorm => normalize each image
            for image, projection in zip(images, projections):
                self.feature_accumulation(projection, image=image)

        else:
            # run all images through 2d cnn together to share batchnorm stats
            # use batchnorm => shape: [B,3,h,w] -> [B*3,h,w] => normalize as batch
            image = images.reshape(images.shape[0]*images.shape[1], *images.shape[2:])
            image = self.normalizer(image)

            tmp = None
            for i in range(image.shape[0]):
                sub_image = image[i*1:(i+1)*1].to(self.device)
                features = self.backbone2d(sub_image)
                if tmp is None:
                    tmp = features
                else:
                    tmp = torch.cat([tmp, features], dim=0)
            features = tmp

            # reshape back
            features = features.view(images.shape[0],
                                    images.shape[1],
                                    *features.shape[1:])
        
            for projection, feature in zip(projections, features):
                self.feature_accumulation(projection, feature=feature)

        # run 3d cnn
        outputs3d, losses3d = self.refine_3d_cnn(targets3d)

        #return {**outputs2d, **outputs3d}, {**losses2d, **losses3d}
        return outputs3d, losses3d


    def postprocess(self, batch):
        """ Wraps the network output into a TSDF data structure
        
        Args:
            batch: dict containg network outputs

        Returns:
            list of TSDFs (one TSDF per scene in the batch)
        """
        
        key = 'vol_%02d'%self.voxel_sizes[0] # only get vol of final resolution
        out = []
        batch_size = len(batch[key+'_tsdf'])

        for i in range(batch_size):
            tsdf = TSDF(self.voxel_size, 
                        self.origin,
                        batch[key+'_tsdf'][i].squeeze(0))

            # add semseg vol
            if ('semseg' in self.voxel_types) and (key+'_semseg' in batch):
                semseg = batch[key+'_semseg'][i]
                if semseg.ndim==4:
                    semseg = semseg.argmax(0)
                tsdf.attribute_vols['semseg'] = semseg

            # add color vol
            if 'color' in self.voxel_types:
                color = batch[key+'_color'][i]
                tsdf.attribute_vols['color'] = color
            out.append(tsdf)

        return out


    def get_transform(self, is_train):
        """ Gets a transform to preprocess the input data"""

        if is_train:
            voxel_dim = self.voxel_dim_train
            random_rotation = self.cfg.DATA.RANDOM_ROTATION_3D
            random_translation = self.cfg.DATA.RANDOM_TRANSLATION_3D
            paddingXY = self.cfg.DATA.PAD_XY_3D
            paddingZ = self.cfg.DATA.PAD_Z_3D
        else:
            # center volume
            voxel_dim = self.voxel_dim_val
            random_rotation = False
            random_translation = False
            paddingXY = 0
            paddingZ = 0

        transform = []
        transform += [transforms.ResizeImage((640,480)),
                      transforms.ToTensor(),
                      transforms.InstanceToSemseg('nyu40'),
                      transforms.RandomTransformSpace(
                          voxel_dim, random_rotation, random_translation,
                          paddingXY, paddingZ),
                      transforms.FlattenTSDF(),
                      transforms.IntrinsicsPoseToProjection(),
                     ]

        return transforms.Compose(transform)


    def train_dataloader(self):
        transform = self.get_transform(True)
        info_files = parse_splits_list(self.cfg.DATASETS_TRAIN)
        dataset = ScenesDataset(
            info_files, self.num_frames_train, transform,
            self.frame_types, self.frame_selection, self.voxel_types,
            self.voxel_sizes)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size_train, num_workers=self.cfg.DATA.NUM_WORKERS, pin_memory=True,
            collate_fn=collate_fn, shuffle=False, drop_last=True, sampler=train_sampler)
        return dataloader

    def val_dataloader(self):
        transform = self.get_transform(False)
        info_files = parse_splits_list(self.cfg.DATASETS_VAL)
        dataset = ScenesDataset(
            info_files, self.num_frames_val, transform,
            self.frame_types, self.frame_selection, self.voxel_types,
            self.voxel_sizes)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=1, collate_fn=collate_fn,
            shuffle=False, drop_last=False)
        return dataloader


    def training_step(self, batch, batch_idx):
        outputs, losses = self.forward(batch)
        
        # visualize training outputs at the begining of each epoch
        if batch_idx==0:
            pred_tsdfs = self.postprocess(outputs)
            trgt_tsdfs = self.postprocess(batch)
            self.logger.experiment1.save_mesh(pred_tsdfs[0], 'train_pred.ply')
            self.logger.experiment1.save_mesh(trgt_tsdfs[0], 'train_trgt.ply')
            
        loss = sum(losses.values())
        return {'loss': loss, 'log': losses}

    def validation_step(self, batch, batch_idx):
        outputs, losses = self.forward(batch)

        # save validation meshes
        pred_tsdfs = self.postprocess(outputs)
        trgt_tsdfs = self.postprocess(batch)
        self.logger.experiment1.save_mesh(pred_tsdfs[0],
                                          batch['scene'][0]+'_pred.ply')
        self.logger.experiment1.save_mesh(trgt_tsdfs[0], 
                                          batch['scene'][0]+'_trgt.ply')

        return losses

    def validation_epoch_end(self, outputs):
        avg_losses = {'val_'+key:torch.stack([x[key] for x in outputs]).mean() 
                      for key in outputs[0].keys()}
        avg_loss = sum(avg_losses.values())
        return {'val_loss': avg_loss, 'log': avg_losses}


    def configure_optimizers(self):
        # optimizers = []
        # schedulers = []

        # allow for different learning rates between pretrained layers 
        # (resnet backbone) and new layers (everything else).
        params_backbone2d = self.backbone2d[0].parameters()
        modules_rest = [self.backbone2d[1], self.backbone3d,
                        self.heads2d, self.heads3d]
        params_rest = itertools.chain(*(params.parameters() 
                                        for params in modules_rest))
        
        # optimzer
        if self.cfg.OPTIMIZER.TYPE == 'Adam':
            lr = self.cfg.OPTIMIZER.ADAM.LR
            lr_backbone2d = lr * self.cfg.OPTIMIZER.BACKBONE2D_LR_FACTOR
            optimizer = torch.optim.Adam([
                {'params': params_backbone2d, 'lr': lr_backbone2d},
                {'params': params_rest, 'lr': lr}])

        else:
            raise NotImplementedError(
                'optimizer %s not supported'%self.cfg.OPTIMIZER.TYPE)

        # scheduler
        if self.cfg.SCHEDULER.TYPE == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.cfg.SCHEDULER.STEP_LR.STEP_SIZE,
                gamma=self.cfg.SCHEDULER.STEP_LR.GAMMA)

        elif self.cfg.SCHEDULER.TYPE != 'None':
            raise NotImplementedError(
                'optimizer %s not supported'%self.cfg.OPTIMIZER.TYPE)
                
        return optimizer, scheduler