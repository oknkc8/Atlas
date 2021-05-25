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

import pytorch_lightning as pl
import torch
from torch import nn
import torch.distributed as dist

from atlas.config import get_parser, get_cfg
from atlas.logger_pytorch import AtlasLogger
from atlas.model_pytorch import VoxelNet

from tqdm import tqdm
import datetime

def main_worker(gpu, ngpus_per_node, cfg):
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=gpu)

    model = VoxelNet(cfg.convert_to_dict())

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    result_folder_name = timestamp + '(' + cfg.TRAINER.NAME + '_' + cfg.TRAINER.VERSION + ')'

    #save_path = os.path.join(cfg.LOG_DIR, cfg.TRAINER.NAME, cfg.TRAINER.VERSION)
    save_path = os.path.join(cfg.LOG_DIR, result_folder_name)
    checkpoint_path = os.path.join(save_path, 'ckpts')
    os.makedirs(checkpoint_path, exist_ok=True)

    logger = AtlasLogger(save_path)

    # Set Dataloader & Optimizer
    train_loader = model.train_dataloader()
    val_loader = model.val_dataloader()

    # Set Optimizer
    optimizer, scheduler = model.configure_optimizers()

    #model = nn.DataParallel(model)
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    valid_min_loss = 1e10

    dist.barrier()
    for epoch in range(cfg.TRAINER.EPOCH):

        print('Training...')
        model.train()
        for i, batch in tqdm(enumerate(train_loader)):
            outputs, losses = model.forward(batch)

            total_loss = 0
            for key, loss in losses.items():
                total_loss += loss
                logger.loss_writer.add_scalar('train_' + key, loss, len(train_loader)*epoch + i+1)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (i+1) % cfg.TRAINER.LOG_STEP == 0:
                print("Epoch:%3d Iter:(%3d/%3d) Total: %.5f" % (epoch+1, i+1, len(train_loader), total_loss), end=' ')
                for key, loss in losses.items():
                    print("%s: %.5f" % (key, loss), end=' ')
                print()

                pred_tsdfs = model.module.postprocess(outputs)
                trgt_tsdfs = model.module.postprocess(batch)

                logger.mesh_writer.save_mesh(pred_tsdfs[0], '%03d_%03d_train_pred.ply'%(epoch+1, i+1))
                logger.mesh_writer.save_mesh(trgt_tsdfs[0], '%03d_%03d_train_trgt.ply'%(epoch+1, i+1))
        
        print('\nValidating...')
        model.eval()
        loss_avg = {}
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_loader)):
                outputs, losses = model.forward(batch)

                total_loss = 0
                for key, loss in losses.items():
                    total_loss += loss
                    if loss_avg.get(key) == None:
                        loss_avg[key] = total_loss
                    else:
                        loss_avg[key] += total_loss

                    logger.loss_writer.add_scalar('val_' + key, loss, len(val_loader)*epoch + i+1)
                
                if loss_avg.get('total_loss') == None:
                    loss_avg['total_loss'] = total_loss
                else:
                    loss_avg['total_loss'] += total_loss
                
                print("Epoch:%3d Iter:(%3d/%3d) Total: %.5f" % (epoch+1, i+1, len(train_loader), total_loss.item()), end=' ')
                for key, loss in losses.items():
                    print("%s: %.5f" % (key, loss.item()), end=' ')
                print()

                pred_tsdfs = model.module.postprocess(outputs)
                trgt_tsdfs = model.module.postprocess(batch)

                logger.mesh_writer.save_mesh(pred_tsdfs[0], ('%03d_%03d_'%(epoch+1, i+1)) + batch['scene'][0]+'_pred.ply')
                logger.mesh_writer.save_mesh(trgt_tsdfs[0], ('%03d_%03d_'%(epoch+1, i+1)) + batch['scene'][0]+'_trgt.ply')
            
            print("[Valid Avg] Epoch:%3d" % (epoch+1), end=' ')
            for key, loss in loss_avg.items():
                loss_avg[key] /= len(val_loader)
                print("%s: %.5f" % (key, loss_avg[key].item()), end=' ')
            print('\n')

        if loss_avg['total_loss'].item() < valid_min_loss:
            valid_min_loss = loss_avg['total_loss'].item()
            
            torch.save({
                    'model' : model.state_dict(),
                    'epoch' : epoch,
                    'cfg' : model.cfg
                },
                os.path.join(checkpoint_path, 'best_model.ckpt')
            )

        torch.save({
                'model' : model.state_dict(),
                'epoch' : epoch,
                'cfg' : model.cfg
            },
            os.path.join(checkpoint_path, str(epoch+1) + '.ckpt')
        )

if __name__ == "__main__":
    args = get_parser().parse_args()

    cfg = get_cfg(args)
    gpus = ''
    for n in cfg.TRAINER.GPUS:
        gpus += str(n)
        gpus += ','
    gpus = gpus[:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ngpus_per_node = len(cfg.TRAINER.GPUS)
    world_size = ngpus_per_node
 
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, ))