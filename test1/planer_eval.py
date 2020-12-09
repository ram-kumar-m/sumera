import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import sys
import os
import cv2
import copy
import glob

from planer_models.model import *
from planer_models.refinement_net import RefineModel
from planer_models.modules import *

class PlaneRCNNDetector():
    def __init__(self, options, config, modelType, checkpoint_dir=''):
        self.options = options
        self.config = config
        self.modelType = modelType
        self.model = MaskRCNN(config)
        self.model.cuda()
        self.model.eval()

        if modelType == 'basic':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/pair_' + options.anchorType
        elif modelType == 'pair':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/pair_' + options.anchorType
        elif modelType == 'refine':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/instance_' + options.anchorType
        elif modelType == 'refine_single':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/refinement_' + options.anchorType
        elif modelType == 'occlusion':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/plane_' + options.anchorType
        elif modelType == 'final':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/planercnn_' + options.anchorType
            pass

        if options.suffix != '':
            checkpoint_dir += '_' + options.suffix
            pass

        ## Indicates that the refinement network is trained separately        
        separate = modelType == 'refine'

        if not separate:
            if options.startEpoch >= 0:
                self.model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_' + str(options.startEpoch) + '.pth'))
            else:
                self.model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint.pth'))
                pass
            pass

        if 'refine' in modelType or 'final' in modelType:
            self.refine_model = RefineModel(options)

            self.refine_model.cuda()
            self.refine_model.eval()
            if not separate:
                state_dict = torch.load(checkpoint_dir + '/checkpoint_refine.pth')
                self.refine_model.load_state_dict(state_dict)
                pass
            else:
                self.model.load_state_dict(torch.load('checkpoint/pair_' + options.anchorType + '_pair/checkpoint.pth'))
                self.refine_model.load_state_dict(torch.load('checkpoint/instance_normal_refine_mask_softmax_valid/checkpoint_refine.pth'))
                pass
            pass

        return

    def detect(self, sample):

        input_pair = []
        detection_pair = []
        camera = sample[30][0].cuda()
        for indexOffset in [0, ]:
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, depth_np_pred = self.model.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='inference_detection', use_nms=2, use_refinement=True)

            if len(detections) > 0:
                detections, detection_masks = unmoldDetections(self.config, camera, detections, detection_masks, depth_np_pred, debug=False)
                pass

            XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(self.config, camera, detections, detection_masks, depth_np_pred, return_individual=True)
            detection_mask = detection_mask.unsqueeze(0)

            input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera})

            if 'nyu_dorn_only' in self.options.dataset:
                XYZ_pred[1:2] = sample[27].cuda()
                pass

            detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'depth_np': depth_np_pred, 'plane_XYZ': plane_XYZ})
            continue

        if ('refine' in self.modelType or 'refine' in self.options.suffix):
            pose = sample[26][0].cuda()
            pose = torch.cat([pose[0:3], pose[3:6] * pose[6]], dim=0)
            pose_gt = torch.cat([pose[0:1], -pose[2:3], pose[1:2], pose[3:4], -pose[5:6], pose[4:5]], dim=0).unsqueeze(0)
            camera = camera.unsqueeze(0)

            for c in range(1):
                detection_dict, input_dict = detection_pair[c], input_pair[c]

                new_input_dict = {k: v for k, v in input_dict.items()}
                new_input_dict['image'] = (input_dict['image'] + self.config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))) / 255.0 - 0.5
                new_input_dict['image_2'] = (sample[13].cuda() + self.config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))) / 255.0 - 0.5
                detections = detection_dict['detection']
                detection_masks = detection_dict['masks']
                depth_np = detection_dict['depth_np']
                image = new_input_dict['image']
                image_2 = new_input_dict['image_2']
                depth_gt = new_input_dict['depth'].unsqueeze(1)

                masks_inp = torch.cat([detection_masks.unsqueeze(1), detection_dict['plane_XYZ']], dim=1)

                segmentation = new_input_dict['segmentation']

                detection_masks = torch.nn.functional.interpolate(detection_masks[:, 80:560].unsqueeze(1), size=(192, 256), mode='nearest').squeeze(1)
                image = torch.nn.functional.interpolate(image[:, :, 80:560], size=(192, 256), mode='bilinear')
                image_2 = torch.nn.functional.interpolate(image_2[:, :, 80:560], size=(192, 256), mode='bilinear')
                masks_inp = torch.nn.functional.interpolate(masks_inp[:, :, 80:560], size=(192, 256), mode='bilinear')
                depth_np = torch.nn.functional.interpolate(depth_np[:, 80:560].unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1)
                plane_depth = torch.nn.functional.interpolate(detection_dict['depth'][:, 80:560].unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1)
                segmentation = torch.nn.functional.interpolate(segmentation[:, 80:560].unsqueeze(1).float(), size=(192, 256), mode='nearest').squeeze().long()

                new_input_dict['image'] = image
                new_input_dict['image_2'] = image_2

                results = self.refine_model(image, image_2, camera, masks_inp, detection_dict['detection'][:, 6:9], plane_depth, depth_np)

                masks = results[-1]['mask'].squeeze(1)

                all_masks = torch.softmax(masks, dim=0)

                masks_small = all_masks[1:]
                all_masks = torch.nn.functional.interpolate(all_masks.unsqueeze(1), size=(480, 640), mode='bilinear').squeeze(1)
                all_masks = (all_masks.max(0, keepdim=True)[1] == torch.arange(len(all_masks)).cuda().long().view((-1, 1, 1))).float()
                masks = all_masks[1:]
                detection_masks = torch.zeros(detection_dict['masks'].shape).cuda()
                detection_masks[:, 80:560] = masks


                detection_dict['masks'] = detection_masks
                detection_dict['depth_ori'] = detection_dict['depth'].clone()
                detection_dict['mask'][:, 80:560] = (masks.max(0, keepdim=True)[0] > (1 - masks.sum(0, keepdim=True))).float()

                if self.options.modelType == 'fitting':
                    masks_cropped = masks_small
                    ranges = self.config.getRanges(camera).transpose(1, 2).transpose(0, 1)
                    XYZ = torch.nn.functional.interpolate(ranges.unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1) * results[-1]['depth'].squeeze(1)
                    detection_areas = masks_cropped.sum(-1).sum(-1)
                    A = masks_cropped.unsqueeze(1) * XYZ
                    b = masks_cropped
                    Ab = (A * b.unsqueeze(1)).sum(-1).sum(-1)
                    AA = (A.unsqueeze(2) * A.unsqueeze(1)).sum(-1).sum(-1)
                    plane_parameters = torch.stack([torch.matmul(torch.inverse(AA[planeIndex]), Ab[planeIndex]) if detection_areas[planeIndex] else detection_dict['detection'][planeIndex, 6:9] for planeIndex in range(len(AA))], dim=0)
                    plane_offsets = torch.norm(plane_parameters, dim=-1, keepdim=True)
                    plane_parameters = plane_parameters / torch.clamp(torch.pow(plane_offsets, 2), 1e-4)
                    detection_dict['detection'][:, 6:9] = plane_parameters

                    XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(self.config, camera, detection_dict['detection'], detection_masks, detection_dict['depth'], return_individual=True)
                    detection_dict['depth'] = XYZ_pred[1:2]
                    pass
                continue
            pass
        return detection_pair
