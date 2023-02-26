import sys

import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.common.dtype as mstype

sys.path.append("..")
import utils
import mindspore.numpy as mnp

from model.config import DefaultConfig
import pdb
import time

def coords_fmap2orig(feature, stride):
    """
    transform one feature map coords to orig coords
    Args
    feature [batch_size,h,w,c]
    stride int
    Returns
    coords [n,2]
    """
    h, w = feature.shape[1:3]
    shifts_x = mnp.arange(start=0, stop=w * stride, step=stride)
    shifts_y = mnp.arange(start=0, stop=h * stride, step=stride)
    shift_y, shift_x = mnp.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.transpose()
    shift_y = shift_y.transpose()
    shift_x = mnp.reshape(shift_x, -1)
    shift_y = mnp.reshape(shift_y, -1)
    coords = mnp.stack((shift_x, shift_y), -1) + stride // 2
    return ops.Cast()(coords, mstype.float32)


class GenTargets(nn.Cell):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range
        # assert len(strides) == len(limit_range)

    def construct(self, inputs):
        """
        inputs
        [0]tuple (cls_logits,cnt_logits,reg_preds)
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        [1]gt_boxes [batch_size,m,4]  FloatTensor
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        """
        
        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]
        # cls_targets_all_level = []
        # cnt_targets_all_level = []
        # reg_targets_all_level = []
        cls_targets_all_level = ()
        cnt_targets_all_level = ()
        reg_targets_all_level = ()
        # assert len(self.strides) == len(cls_logits)
        for level in range(len(cls_logits)):
            level_out = (cls_logits[level], cnt_logits[level], reg_preds[level])
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level],self.limit_range[level])                                        
            # cls_targets_all_level.append(level_targets[0])
            # cnt_targets_all_level.append(level_targets[1])
            # reg_targets_all_level.append(level_targets[2])
            cls_targets_all_level = cls_targets_all_level + (level_targets[0],)
            cnt_targets_all_level = cnt_targets_all_level + (level_targets[1],)
            reg_targets_all_level = reg_targets_all_level + (level_targets[2],)

        return ops.Concat(axis=1)(cls_targets_all_level), ops.Concat(axis=1)(cnt_targets_all_level), ops.Concat(axis=1)(
            reg_targets_all_level)

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        '''
        Args
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]
        gt_boxes [batch_size,m,4]
        classes [batch_size,m]
        stride int
        limit_range list [min,max]
        Returns
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits, cnt_logits, reg_preds = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]
        m = gt_boxes.shape[1]

        # assert isinstance(cls_logits, Tensor)
        # assert isinstance(cnt_logits, Tensor)
        # assert isinstance(reg_preds, Tensor)


        

        

        transpose = ops.Transpose()
        reshape = ops.Reshape()

        cls_logits = transpose(cls_logits, (0, 2, 3, 1))  # [batch_size,h,w,class_num]

        coords = coords_fmap2orig(cls_logits, stride)  # [h*w,2]

        cls_logits = reshape(cls_logits, (batch_size, -1, class_num))  # [batch_size,h*w,class_num]


        cnt_logits = transpose(cnt_logits, (0, 2, 3, 1))
        cnt_logits = reshape(cnt_logits, (batch_size, -1, 1))

        reg_preds = transpose(reg_preds, (0, 2, 3, 1))
        reg_preds = reshape(reg_preds, (batch_size, -1, 4))

        h_mul_w = cls_logits.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]

        # assert isinstance(gt_boxes, Tensor)
        # x = Tensor(x, mstype.float32)##
        # y = Tensor(y, mstype.float32)##

        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :] # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]

        ltrb_off = ops.Stack(axis=-1)((l_off, t_off, r_off, b_off))  # [batch_size,h*w,m,4]

        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])  # [batch_size,h*w,m]


        # ltrb_off = Tensor(ltrb_off, mstype.float32)##
        # [batch_size,h*w,m]
        off_min = mnp.amin(ltrb_off, axis=-1)
        # [batch_size,h*w,m]
        off_max = mnp.amax(ltrb_off, axis=-1)

        
        #off_min = ops.ArgMinWithValue(axis=-1)(ltrb_off)[1]
        #off_max = ops.ArgMaxWithValue(axis=-1)(ltrb_off)[1]
        #off_min = off_min.asnumpy()
        #off_max = off_max.asnumpy()

        mask_in_gtboxes = off_min > 0

        tempmin = off_max > limit_range[0]
        tempmax = off_max <= limit_range[1]
        tempmin = ops.Cast()(tempmin,mindspore.int32)
        tempmax = ops.Cast()(tempmax,mindspore.int32)
        tempMask_in_level = ops.Mul()(tempmin,tempmax)
        # mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])
        mask_in_level = ops.Cast()(tempMask_in_level,mindspore.bool_)

        radiu = stride * sample_radiu_ratio
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off = ops.Stack(axis=-1)((c_l_off, c_t_off, c_r_off, c_b_off))  # [batch_size,h*w,m,4]

        # assert isinstance(c_ltrb_off, Tensor)
        #c_ltrb_off.asnumpy()

        c_off_max = mnp.amax(c_ltrb_off, axis=-1)
        # mask_center = Tensor(c_off_max < radiu).asnumpy()
        mask_center = c_off_max < radiu
        # assert isinstance(mask_in_gtboxes, np.ndarray)
        # assert isinstance(mask_in_level, np.ndarray)
        # assert isinstance(mask_center, np.ndarray)

        tempingtboxes = ops.Cast()(mask_in_gtboxes,mindspore.int32)
        tempmaskinlevel = ops.Cast()(mask_in_level,mindspore.int32)
        tempmaskcenter = ops.Cast()(mask_center,mindspore.int32)
      #  mask_pos = mask_in_gtboxes & mask_in_level & mask_center  # [batch_size,h*w,m]
        mask_pos = ops.Mul()(ops.Mul()(tempingtboxes,tempmaskinlevel),tempmaskcenter)

        mask_pos = ops.Cast()(mask_pos, mstype.bool_)

        areas[~mask_pos] = 99999999

        # [batch_size,h*w]
        tempareas = areas.reshape(-1,areas.shape[-1])
        # areas_min_ind = P.ArgMinWithValue(-1)(Tensor(areas.reshape(-1, areas.shape[-1])))
        areas_min_ind = P.ArgMinWithValue(-1)(tempareas)
      #  x = np.arange(0, areas_min_ind[0].shape[0]).astype(np.int32)
        x = mnp.arange(0, areas_min_ind[0].shape[0]).astype(mindspore.int32)
        # indices = P.Concat(-1)((P.ExpandDims()(Tensor(x), -1), P.ExpandDims()(areas_min_ind[0], -1)))
        indices = P.Concat(-1)((P.ExpandDims()(x, -1), P.ExpandDims()(areas_min_ind[0], -1)))
        # reg_targets = P.GatherNd()(Tensor(ltrb_off.reshape(-1, m, 4)), indices)
        reg_targets = P.GatherNd()(ltrb_off.reshape(-1, m, 4), indices)
        reg_targets = ops.Reshape()(reg_targets,(batch_size,-1,4))


        # classes = ops.BroadcastTo(areas.shape)(classes[:,None,:])
        classes = mnp.broadcast_to(classes[:, None, :], areas.shape)
        # cls_targets = P.GatherNd()(Tensor(classes.reshape(-1,m)),indices)
        cls_targets = P.GatherNd()(classes.reshape(-1, m), indices)
        cls_targets = ops.Reshape()(cls_targets,(batch_size,-1,1))


        # [batch_size,h*w]
        left_right_min = ops.Minimum()(reg_targets[..., 0], reg_targets[..., 2])
        left_right_max = ops.Maximum()(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = ops.Minimum()(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = ops.Maximum()(reg_targets[..., 1], reg_targets[..., 3])

        # [batch_size,h*w,1]
        # cnt_targets = ops.Sqrt()((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-8))
        cnt_targets = ops.Sqrt()((left_right_min * top_bottom_min + 1e-8) / (left_right_max * top_bottom_max + 1e-8))
        cnt_targets = ops.ExpandDims()(cnt_targets, -1)

        # assert reg_targets.shape == (batch_size, h_mul_w, 4)
        # assert cls_targets.shape == (batch_size, h_mul_w, 1)
        # assert cnt_targets.shape == (batch_size, h_mul_w, 1)

        # process neg coords
        mask_pos_2 = ops.Cast()(mask_pos,mstype.float16)
        mask_pos_2 = ops.ReduceSum()(mask_pos_2, -1)
        mask_pos_2 = mask_pos_2 >= 1
        # assert mask_pos_2.shape == (batch_size, h_mul_w)
        # mask_pos_2 = mask_pos_2.asnumpy()
        # cls_targets = cls_targets.asnumpy()
        # cnt_targets = cnt_targets.asnumpy()
        # reg_targets = reg_targets.asnumpy()

        expand_dims = ops.ExpandDims()
        mask_pos_2 = expand_dims(mask_pos_2,2)
        cls_targets[~mask_pos_2] = 0  # [batch_size,h*w,1]
        cnt_targets[~mask_pos_2] = -1

        stack = ops.Stack(axis=2)
        tempmask = ()
        for i in range(4):
            tempmask += (mask_pos_2,)
        mask_pos_2 = stack(tempmask)

        squeeze = ops.Squeeze(3)
        mask_pos_2 = squeeze(mask_pos_2)

        reg_targets[~mask_pos_2] = -1

        # cls_targets = Tensor(cls_targets, mstype.float32)
        # cnt_targets = Tensor(cnt_targets, mstype.float32)
        # reg_targets = Tensor(reg_targets, mstype.float32)

        return cls_targets, cnt_targets, reg_targets


class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)


def compute_cls_loss(preds, targets, mask):
    '''
    Args
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size = targets.shape[0]
    preds_reshape = ()
    class_num = preds[0].shape[1]
    mask = ops.ExpandDims()(mask, -1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    # [batch_size,]
    mask = ops.Cast()(mask, mstype.float32)
    num_pos = ops.ReduceSum()(mask, (1, 2))
    # min_value = Tensor(1, mstype.float32)
    # max_value = Tensor(sys.maxsize, mstype.float32)
    ones = ops.Ones()
    candidate = ones(num_pos.shape, mindspore.float32)
    num_pos = mnp.where(num_pos==0,candidate,num_pos)
   # num_pos = ops.clip_by_value(num_pos, min_value, max_value)    #change1
    num_pos = ops.Cast()(num_pos, mstype.float32)
    for pred in preds:
        pred = ops.Transpose()(pred, (0, 2, 3, 1))
        pred = ops.Reshape()(pred, (batch_size, -1, class_num))
        preds_reshape = preds_reshape + (pred,)
    # [batch_size,sum(_h*_w),class_num]
    preds = ops.Concat(axis=1)(preds_reshape)
    # assert preds.shape[:2] == targets.shape[:2]
    loss = ()
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]  # [sum(_h*_w),class_num]
        target_pos = targets[batch_index]  # [sum(_h*_w),1]
        # ar = np.arange(1, class_num + 1)
        # ar = Tensor(ar[None, :], mstype.float32)
        ar = mnp.arange(1, class_num + 1).astype(mstype.float32)
        ar = ar[None, :]
        target_pos = (ar == target_pos)
        # sparse-->onehot
        target_pos = ops.Cast()(target_pos, mstype.float32)
        fl_result = focal_loss_from_logits(pred_pos, target_pos)
        fl_result = ops.Reshape()(fl_result, (1,))
     #   fl_result = Tensor(fl_result, mstype.float32)
        loss = loss + (fl_result,)
    # [batch_size,]
    return ops.Concat()(loss) / num_pos


# def compute_cnt_loss(preds, targets, mask):
#     '''
#     Args
#     preds: list contains five level pred [batch_size,1,_h,_w]
#     targets: [batch_size,sum(_h*_w),1]
#     mask: [batch_size,sum(_h*_w)]:Tensor(Bool)
#     '''
#
#     min_value = Tensor(1, mstype.float32)
#     max_value = Tensor(sys.maxsize, mstype.float32)
#
#     batch_size = targets.shape[0]
#     c = targets.shape[-1]
#     preds_reshape = ()
#     mask = ops.ExpandDims()(mask, -1)
#
#     mask = ops.Cast()(mask, mstype.float32)
#     num_pos = ops.ReduceSum()(mask, axis=[1, 2])
#     num_pos = ops.clip_by_value(num_pos, min_value, max_value)
#     num_pos = ops.Cast()(num_pos, mstype.float32)
#     for pred in preds:
#         pred = P.Transpose()(pred, (0, 2, 3, 1))
#         pred = P.Reshape()(pred, (batch_size, -1, c))
#         preds_reshape = preds_reshape + (pred,)
#     preds = P.Concat(axis=1)(preds_reshape)
#     assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),1]
#     loss = ()
#     for batch_index in range(batch_size):
#
#         mask_index = mask[batch_index]
#         mask_index = ops.Cast()(mask_index, mstype.int32)
#
#         s = time.time()
#         ###GET INDEX AND FLAG###
#         final_index = []
#         noEmpty = False
#         count = 0
#         temp_mask = mask_index.flatten()
#
#         for i in temp_mask:
#             if i>0:
#                 final_index.append(count)
#                 noEmpty = True
#             count +=1
#         ############################
#         e = time.time()
#         print("time",e-s)
#
#         if noEmpty:
#             ####get pred_pos######
#             unsqueeze = ops.ExpandDims()
#             pred_pos = unsqueeze(preds[batch_index].flatten()[final_index], -1)
#             target_pos = unsqueeze(targets[batch_index].flatten()[final_index], -1)
#             #   pred_pos = preds[batch_index][mask_index]
#             #   target_pos = targets[batch_index][mask_index]
#             # pred_pos = P.Squeeze()(pred_pos)
#             # target_pos = P.Squeeze()(target_pos)
#             pred_pos = P.Squeeze(1)(pred_pos)
#             target_pos = P.Squeeze(1)(target_pos)
#             assert len(pred_pos.shape) == 1
#
#             weight = P.Ones()(pred_pos.shape, mstype.float32)
#
#             pred_pos = P.Sigmoid()(pred_pos)
#             bce_result = P.BinaryCrossEntropy(reduction='sum')(pred_pos, target_pos, weight)
#             bce_result = P.Reshape()(bce_result, (1,))
#         else:
#             bce_result = Tensor([0.])
#         loss += (bce_result,)
#     return P.Concat(axis=0)(loss) / num_pos

def compute_cnt_loss(preds, targets, mask):
    '''
    Args
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]:Tensor(Bool)
    '''

    # min_value = Tensor(1, mstype.float32)
    # max_value = Tensor(sys.maxsize, mstype.float32)

    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = ()
    mask = ops.ExpandDims()(mask, -1)

    mask = ops.Cast()(mask, mstype.float32)
    num_pos = ops.ReduceSum()(mask, (1, 2))
   # num_pos = ops.clip_by_value(num_pos, min_value, max_value)
    ones = ops.Ones()
    candidate = ones(num_pos.shape, mindspore.float32)
    num_pos = mnp.where(num_pos==0,candidate,num_pos)

    num_pos = ops.Cast()(num_pos, mstype.float32)
    for pred in preds:
        pred = P.Transpose()(pred, (0, 2, 3, 1))
        pred = P.Reshape()(pred, (batch_size, -1, c))
        preds_reshape = preds_reshape + (pred,)
    preds = P.Concat(axis=1)(preds_reshape)
    # assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),1]
    loss = ()
    for batch_index in range(batch_size):

        # mask_index = mask[batch_index]
        # mask_index = ops.Cast()(mask_index, mstype.bool_)
        # masked_select = ops.MaskedSelect()

        # ones = P.Ones()
        # dmask_index  = ones(mask_index.shape,mstype.bool_)

        # pred_pos = masked_select(preds[batch_index].flatten(), dmask_index.flatten())
        # target_pos = masked_select(targets[batch_index].flatten(), dmask_index.flatten())

        # assert len(pred_pos.shape) == 1
        pred_pos = preds[batch_index].flatten()
        target_pos = targets[batch_index].flatten()

        weight = P.Ones()(pred_pos.shape, mstype.float32)
        pred_pos = P.Sigmoid()(pred_pos)

        if pred_pos.shape[0] != 0:
          #  print("cnt dynamic shape:", pred_pos.shape, target_pos.shape, weight.shape)
            bce_result = P.BinaryCrossEntropy(reduction='none')(pred_pos, target_pos, weight)
         #   bce_result = nn.BCEWithLogitsLoss(reduction='none')(pred_pos, target_pos)
            bce_result = ops.dot(bce_result.reshape(1, -1), mask[batch_index])

        else:
            # bce_result = Tensor([0.])
            bce_result = mnp.zeros((1,),mindspore.float32)
        bce_result = P.Reshape()(bce_result, (1,))

        loss += (bce_result,)

    return P.Concat(axis=0)(loss) / num_pos


def compute_reg_loss(preds, targets, mask, mode='giou'):
    '''
    Args
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''

    # mask = ops.Ones()(mask.shape,mindspore.float32)
    # mask = mask>0

    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = ()
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    mask = ops.ExpandDims()(mask, -1)
    mask = ops.Cast()(mask, mstype.float32)
    num_pos = ops.ReduceSum()(mask, (1, 2))
    # min_value = Tensor(1, mstype.float32)
    # max_value = Tensor(sys.maxsize, mstype.float32)
    # num_pos = ops.clip_by_value(num_pos, min_value, max_value)
    ones = ops.Ones()
    candidate = ones(num_pos.shape, mindspore.float32)
    num_pos = mnp.where(num_pos==0,candidate,num_pos)
    num_pos = ops.Cast()(num_pos, mstype.float32)
    for pred in preds:
        pred = ops.Transpose()(pred, (0, 2, 3, 1))
        pred = ops.Reshape()(pred, (batch_size, -1, c))
        preds_reshape = preds_reshape + (pred,)
    preds = ops.Concat(axis=1)(preds_reshape)
    # assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),4]
    loss = ()
    for batch_index in range(batch_size):
        mask_index = mask[batch_index]
        mask_index = ops.Cast()(mask_index, mstype.float32)
       # mask_select = ops.MaskedSelect()

        ##get index
        stack = ops.Stack(axis=0)
        tempmask = ()
        for i in range(preds[batch_index].shape[1]):
            tempmask += (mask_index,)
        mask_index = stack(tempmask)
        squeeze = ops.Squeeze(2)
        mask_index = squeeze(mask_index)
        # mask_index = mask_index.transpose()
        # pred_pos = mask_select(preds[batch_index],mask_index)
        # target_pos = mask_select(targets[batch_index],mask_index)
        pred_pos = preds[batch_index]
        target_pos = targets[batch_index]
        # pred_pos = preds[batch_index][mask_index]  # [num_pos_b,4]
        # target_pos = targets[batch_index][mask_index]  # [num_pos_b,4]
        if pred_pos.shape[0] != 0:
            pred_pos = ops.Reshape()(pred_pos, (-1, 4))
            target_pos = ops.Reshape()(target_pos, (-1, 4))
            # assert len(pred_pos.shape) == 2
            # if mode == 'iou':
            #     loss_result = iou_loss(pred_pos, target_pos)
            # elif mode == 'giou':
            #print("reg dynamic shape:", pred_pos.shape, target_pos.shape, mask_index[0].shape)
            loss_result = giou_loss(pred_pos, target_pos, mask_index[0])
        else:
            #loss_result = Tensor([0.])
            loss_result = mnp.zeros((1,), mindspore.float32)
        loss_result = loss_result.reshape((1,))
        loss = loss + (loss_result,)
    return ops.Concat()(loss) / num_pos  # [batch_size,]


# def iou_loss(preds, targets):
#     '''
#     Args:
#     preds: [n,4] ltrb
#     targets: [n,4]
#     '''
#     minimum = ops.Minimum()
#     lt = minimum(preds[:, :2], targets[:, :2])
#     rb = minimum(preds[:, 2:], targets[:, 2:])
#     max_value = Tensor(sys.maxsize, mstype.float32)
#     # wh = ops.clip_by_value((rb + lt), Tensor(0, mstype.float32), max_value)
#     wh = ops.Abs()(rb + lt)
#     overlap = wh[:, 0] * wh[:, 1]  # [n]
#     area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
#     area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
#     iou = overlap / (area1 + area2 - overlap)
#     # loss = ops.clip_by_value(iou, Tensor(1e-6, mstype.float32), max_value)
#     loss = ops.Abs()(iou)
#     loss = -ops.Log()(loss)
#     loss = Tensor(loss, mstype.float32)
#     return ops.ReduceSum()(loss)


def giou_loss(preds, targets, mask_index):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    minimum = ops.Minimum()
    maximum = ops.Maximum()

    lt_min = minimum(preds[:, :2], targets[:, :2])
    rb_min = minimum(preds[:, 2:], targets[:, 2:])

    # max_value = Tensor(sys.maxsize, mstype.float32)
    # wh_min = ops.clip_by_value((rb_min + lt_min), Tensor(0, mstype.float32), max_value)
    wh_min = ops.Abs()(rb_min + lt_min)
    overlap = wh_min[:, 0] * wh_min[:, 1]  # [n]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    union = (area1 + area2 - overlap)
    iou = overlap / union

    lt_max = maximum(preds[:, :2], targets[:, :2])
    rb_max = maximum(preds[:, 2:], targets[:, 2:])
    # wh_max = ops.clip_by_value((rb_max + lt_max), Tensor(0, mstype.float32), max_value)
    wh_max = ops.Abs()(rb_max + lt_max)
    G_area = wh_max[:, 0] * wh_max[:, 1]  # [n]

    # clamp = ops.clip_by_value(G_area, Tensor(1e-10, mstype.float32), max_value)
    giou = iou - (G_area - union + 1e-8) / ops.Abs()(G_area + 1e-8)    #back3
    loss = (1. - giou).reshape(1, -1)
    mask_index = mask_index.reshape(-1, 1)
    loss = ops.dot(loss, mask_index)
    # loss = Tensor(loss, mstype.float32)
    return loss


def focal_loss_from_logits(preds, targets, gamma=2.0, alpha=0.25):
    '''
    Args:
    preds: [n,class_num]
    targets: [n,class_num]
    '''
    preds = ops.Sigmoid()(preds)
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = -w * ops.Pow()((1.0 - pt), gamma) * ops.Log()(pt+1e-8)    #back2

    return ops.ReduceSum()(loss)


class LOSS(nn.Cell):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def construct(self, inputs):
        '''
        inputs list
        [0]preds:  ....
        [1]targets : list contains three elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        '''
        preds, targets = inputs
        cls_logits, cnt_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets = targets

        # TEST#
        #  cnt_targets = mindspore.numpy.full_like(cnt_targets,-1)
        #######

        mask_pos = ops.Squeeze(axis=-1)(cnt_targets > -1)  # [batch_size,sum(_h*_w)]
        mean = ops.ReduceMean()
        cls_loss = mean(compute_cls_loss(cls_logits, cls_targets, mask_pos))
        cnt_loss = mean(compute_cnt_loss(cnt_logits, cnt_targets, mask_pos))
        reg_loss = mean(compute_reg_loss(reg_preds, reg_targets, mask_pos))
        cls_loss = ops.Reshape()(cls_loss, (1,))
        cnt_loss = ops.Reshape()(cnt_loss, (1,))
        reg_loss = ops.Reshape()(reg_loss, (1,))
       
        total_loss = cls_loss + cnt_loss + reg_loss
        #print("total loss,cls,cnt,reg:",total_loss,cls_loss,cnt_loss,reg_loss)
        return cls_loss, cnt_loss, reg_loss, total_loss

class HM():

        @staticmethod
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = (inputs)
            all_indexes = (indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(
        inputs, indexes, features, mindspore.Tensor([momentum]).to(inputs.device)
    )

class HybridMemoryMultiFocalPercent(nn.Cell):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, top_percent=0.1):
        super(HybridMemoryMultiFocalPercent, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp

        #for mutli focal
        self.top_percent = top_percent

        self.idx = mindspore.ops.zeros(num_memory).long()

        self.register_buffer("features", mindspore.ops.zeros(num_memory, num_features))
        self.register_buffer("labels", mindspore.ops.zeros(num_memory).long())
    
    def _init_ids(self, ids):
        self.idx.data.copy_(ids.long().to(self.labels.device))

    def _update_feature(self, features):
        features = mindspore.ops.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))
    
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    def forward(self, results, indexes):
        inputs = results
        inputs = mindspore.ops.normalize(inputs, p=2, dim=1)

        # inputs: B*2048, features: N*2048
        inputs = hm(inputs, indexes, self.features, self.momentum) #B*N, similarity
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax_multi_focal(vec, mask, dim=1, targets=None, epsilon=1e-6):
            exps = mindspore.ops.exp(vec)
            one_hot_pos = mindspore.ops.nn.functional.one_hot(targets, num_classes=exps.shape[1])
            # assert exps.shape==one_hot_pos.shape
            one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)
            one_hot_neg = one_hot_neg - one_hot_pos
            masked_exps = exps * mask.float().clone()
            neg_exps = exps.new_zeros(size=exps.shape)
            neg_exps[one_hot_neg>0] = masked_exps[one_hot_neg>0]
            ori_neg_exps = neg_exps
            neg_exps = neg_exps/neg_exps.sum(dim=1, keepdim=True)
            new_exps = masked_exps.new_zeros(size=exps.shape)
            new_exps[one_hot_pos>0] = masked_exps[one_hot_pos>0]
            # topk_values, topk_indexes = neg_exps.topk(dim=1, k=self.topk)

            sorted, indices = mindspore.ops.sort(neg_exps, dim=1, descending=True)
            sorted_cum_sum = mindspore.ops.cumsum(sorted, dim=1)
            sorted_cum_diff = (sorted_cum_sum - self.top_percent).abs()
            sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1)
            min_values = sorted[mindspore.ops.range(0, sorted.shape[0]-1).long(), sorted_cum_min_indices]
            min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)
            ori_neg_sum = ori_neg_exps.sum(dim=1, keepdim=True)
            ori_neg_exps[ori_neg_exps<min_values] = 0
            # print((ori_neg_exps/ori_neg_sum).sum(dim=1)[:20])

            new_exps[one_hot_neg>0] = ori_neg_exps[one_hot_neg>0]

            masked_exps = new_exps

            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[indexes].clone()
        labels = self.labels.clone() #shape: N, unique label num: u

        sim = mindspore.ops.zeros(labels.max() + 1, B).float().cuda() #u*B
        sim.index_add_(0, labels, inputs.t().contiguous()) #
        nums = mindspore.ops.zeros(labels.max() + 1, 1).float().cuda() #many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, labels, mindspore.ops.ones(self.num_memory, 1).float().cuda()) #u*1
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) #average features in each cluster, u*B
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax_multi_focal(sim.t().contiguous(), mask.t().contiguous(), targets=targets) #sim: u*B, mask:u*B, masked_sim: B*u
        return mindspore.ops.nll_loss(mindspore.ops.log(masked_sim + 1e-6), targets)


class Quaduplet2Loss(nn.Cell):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, bg_weight=0.25):
        super(Quaduplet2Loss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.bg_weight = bg_weight

    def forward(self, inputs, targets):
        """
        Does not calculate noise inputs with label -1
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        #print(inputs.shape, targets.shape)
        inputs_new = []
        bg = []
        targets_new = []
        targets_value = []
        for i in range(len(targets)):
            if targets[i] < 0:
                bg.append(inputs[i])
            else:
                inputs_new.append(inputs[i])
                targets_new.append(targets[i])
                targets_value.append(targets[i].cpu().numpy().item())
        if len(set(targets_value)) < 2:
            tmp_loss = mindspore.ops.zeros(1)
            tmp_loss = tmp_loss[0]
            tmp_loss = tmp_loss.to(targets.device)
            return tmp_loss
        #print(targets_value)
        inputs_new = mindspore.ops.stack(inputs_new)
        targets_new = mindspore.ops.stack(targets_new)
        #print(inputs_new.shape, targets_new.shape)
        n = inputs_new.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = mindspore.ops.pow(inputs_new, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs_new, inputs_new.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        #print("Triplet ", dist)
        # For each anchor, find the hardest positive and negative
        mask = targets_new.expand(n, n).eq(targets_new.expand(n, n).t())
        #print(mask)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        #dist_ap = torch.cat(dist_ap)
        #dist_an = torch.cat(dist_an)
        dist_ap = mindspore.ops.stack(dist_ap)
        dist_an = mindspore.ops.stack(dist_an)
        # Compute ranking hinge loss
        y = mindspore.ops.ones_like(dist_an)
        #y = dist_an.data.new()
        #y.resize_as_(dist_an.data)
        #y.fill_(1)
        #y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        bg = mindspore.ops.stack(bg)
        m = bg.size(0)
        if m > 0:
            #print("m", m, "n", n)
            dist_p = mindspore.ops.pow(inputs_new, 2).sum(dim=1, keepdim=True).expand(n, m)
            #print("dist_p", dist_p.shape)
            dist_bg = mindspore.ops.pow(bg, 2).sum(dim=1, keepdim=True)
            #print("dist_bg", dist_bg.shape)
            dist_bg = dist_bg.expand(m, n)
            #print("dist_bg", dist_bg.shape)
            dist_new = dist_p + dist_bg.t()
            dist_new.addmm_(1, -2, inputs_new, bg.t())
            dist_new = dist_new.clamp(min=1e-12).sqrt()  # for numerical stability
            # For each anchor, find the hardest positive and negative
            #mask = targets_new.expand(n, ).eq(targets_new.expand(n, n).t())
            dist_ap, dist_an = [], []
            for i in range(n):
                #dist_ap.append(dist[i][mask[i]].max())
                dist_ap.append(dist[i].max())
                dist_an.append(dist_new[i].min())
            dist_ap = mindspore.ops.stack(dist_ap)
            dist_an = mindspore.ops.stack(dist_an)
            y = mindspore.ops.ones_like(dist_an)
            loss_bg = self.ranking_loss(dist_an, dist_ap, y)
        loss += self.bg_weight*loss_bg

        return loss

if __name__ == "__main__":
    ones = P.Ones()
    loss1 = compute_cls_loss([ones((2, 1, 4, 4), mstype.float32)] * 5, ones((2, 80, 1), mstype.float32), ones((2, 80), mstype.bool_))
    loss2 = compute_cnt_loss([ones((2, 1, 4, 4), mstype.float32)] * 5, ones((2, 80, 1), mstype.float32),ones((2, 80), mstype.bool_))
    loss3 = compute_reg_loss([ones((2, 4, 4, 4), mstype.float32)] * 5, ones((2, 80, 4), mstype.float32),
                            ones((2, 80), mstype.bool_))
    print(loss1,loss2,loss3)