'''
Taken from https://github.com/Sxela/flow_tools/blob/main
Minimal changes have been made, some code removed, some commented. Check the link for the original code(s).
'''
import numpy as np

def make_consistency(flow1, flow2, edges_unreliable=False):
      # Awesome pythonic consistency check from [maua](https://github.com/maua-maua-maua/maua/blob/44485c745c65cf9d83cb1b1c792a177588e9c9fc/maua/flow/consistency.py) by Hans Brouwer and Henry Rachootin
      # algorithm based on https://github.com/manuelruder/artistic-videos/blob/master/consistencyChecker/consistencyChecker.cpp
      # reimplemented in numpy by Hans Brouwer
      # // consistencyChecker
      # // Check consistency of forward flow via backward flow.
      # // (c) Manuel Ruder, Alexey Dosovitskiy, Thomas Brox 2016

      flow1 = np.flip(flow1, axis=2)
      flow2 = np.flip(flow2, axis=2)
      h, w, _ = flow1.shape

      # get grid of coordinates for each pixel
      orig_coord = np.flip(np.mgrid[:w, :h], 0).T

      # find where the flow1 maps each pixel
      warp_coord = orig_coord + flow1

      # clip the coordinates in bounds and round down
      warp_coord_inbound = np.zeros_like(warp_coord)
      warp_coord_inbound[..., 0] = np.clip(warp_coord[..., 0], 0, h - 2)
      warp_coord_inbound[..., 1] = np.clip(warp_coord[..., 1], 0, w - 2)
      warp_coord_floor = np.floor(warp_coord_inbound).astype(int)

      # for each pixel: bilinear interpolation of the corresponding flow2 values around the point mapped to by flow1
      alpha = warp_coord_inbound - warp_coord_floor
      flow2_00 = flow2[warp_coord_floor[..., 0], warp_coord_floor[..., 1]]
      flow2_01 = flow2[warp_coord_floor[..., 0], warp_coord_floor[..., 1] + 1]
      flow2_10 = flow2[warp_coord_floor[..., 0] + 1, warp_coord_floor[..., 1]]
      flow2_11 = flow2[warp_coord_floor[..., 0] + 1, warp_coord_floor[..., 1] + 1]
      flow2_0_blend = (1 - alpha[..., 1, None]) * flow2_00 + alpha[..., 1, None] * flow2_01
      flow2_1_blend = (1 - alpha[..., 1, None]) * flow2_10 + alpha[..., 1, None] * flow2_11
      warp_coord_flow2 = (1 - alpha[..., 0, None]) * flow2_0_blend + alpha[..., 0, None] * flow2_1_blend

      # coordinates that flow2 remaps each flow1-mapped pixel to
      rewarp_coord = warp_coord + warp_coord_flow2

      # where the difference in position after flow1 and flow2 are applied is larger than a threshold there is likely an
      # occlusion. set values to -1 so the final gaussian blur will spread the value a couple pixels around this area
      squared_diff = np.sum((rewarp_coord - orig_coord) ** 2, axis=2)
      threshold = 0.01 * np.sum(warp_coord_flow2 ** 2 + flow1 ** 2, axis=2) + 0.5
      
      reliable_flow = np.ones((squared_diff.shape[0], squared_diff.shape[1], 3))
      reliable_flow[...,0] = np.where(squared_diff >= threshold, -0.75, 1)

      # areas mapping outside of the frame are also occluded (don't need extra region around these though, so set 0)
      if edges_unreliable:
          reliable_flow[...,1] = np.where(
              np.logical_or.reduce(
                  (
                      warp_coord[..., 0] < 0,
                      warp_coord[..., 1] < 0,
                      warp_coord[..., 0] >= h - 1,
                      warp_coord[..., 1] >= w - 1,
                  )
              ),
              0,
              reliable_flow[...,1],
          )

      # get derivative of flow, large changes in derivative => edge of moving object
      dx = np.diff(flow1, axis=1, append=0)
      dy = np.diff(flow1, axis=0, append=0)
      motion_edge = np.sum(dx ** 2 + dy ** 2, axis=2)
      motion_threshold = 0.01 * np.sum(flow1 ** 2, axis=2) + 0.002
      reliable_flow[...,2] = np.where(np.logical_and(motion_edge > motion_threshold, reliable_flow[...,2] != -0.75), 0, reliable_flow[...,2])

      return reliable_flow