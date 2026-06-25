import numpy as np
from dendroptimized import voxelize


class RecursiveMainXYAxisTilingMask:
    """Recursively partitions a point cloud into tiles using principal directional splits.

    At each level, points are splitted along their principal XY direction (computed by PCA).

    Highly Inspired / Adapted from Superpoint Transformer / Damien Robert:
    https://github.com/drprojects/superpoint_transformer/blob/eb959b61226f60e0c037cc105c56d51318650e8a/src/transforms/sampling.py#L571
    """

    def __init__(self, tiling_factor: int = 2, voxel_size: float = 0.20) -> None:
        assert tiling_factor >= 0
        self.tiling_factor = tiling_factor
        self.voxel_size = voxel_size
        self.global_indicator: np.ndarray

    def tile(
        self,
        point_cloud: np.ndarray,
    ) -> np.ndarray:
        if self.tiling_factor == 0:
            self.global_indicator = np.ones(point_cloud.shape[0], dtype=np.int64)
        else:
            self.global_indicator = np.zeros(point_cloud.shape[0], dtype=np.int64)
            cumulative_mask = np.ones(point_cloud.shape[0], dtype=bool)
            self._full_mask(point_cloud, 0, 0, cumulative_mask)
        return self.global_indicator

    def _full_mask(
        self,
        point_cloud: np.ndarray,
        cur_step: int,
        cur_id: int,
        cum_binary_mask: np.ndarray,
    ) -> None:
        min_id_level = 2**cur_step - 1
        id_in_level = cur_id - min_id_level
        if cur_step == self.tiling_factor:
            self.global_indicator[cum_binary_mask] = id_in_level + 1
            return
        else:
            next_id_base = 2**cur_step + cur_id + id_in_level
            left_mask, right_mask = self._split_by_main_xy_direction(point_cloud, cum_binary_mask)
            # recursive call to full_mask
            self._full_mask(point_cloud, cur_step + 1, next_id_base, left_mask)
            self._full_mask(point_cloud, cur_step + 1, next_id_base + 1, right_mask)

    def _split_by_main_xy_direction(self, point_cloud: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        current_tile = point_cloud[mask, :]
        v = self._compute_main_xy_direction(current_tile)
        if v[0] < 0.0:
            v = -v

        # Project points along this direction and split around the median
        proj = point_cloud[:, :2] @ v  # faster and clearer
        current_tile_proj = np.sum(current_tile[:, :2] * v, axis=1)

        new_mask = proj < np.median(current_tile_proj)

        # Compute right and left masks
        left_mask = mask & new_mask  # accumulate & mask
        right_mask = mask & ~new_mask  # accumulate & ~mask
        return left_mask, right_mask

    def _compute_main_xy_direction(self, point_cloud: np.ndarray) -> np.ndarray:

        # Voxelize : TODO voxelize once
        voxelated_cloud, _, _ = voxelize(
            point_cloud.astype(np.float64),
            self.voxel_size,
            self.voxel_size,
            5,
            with_n_points=False,
            verbose=False,
        )

        # Principal component on XY with SVD
        xy_voxelated = voxelated_cloud[:, :2]
        xy_centered = xy_voxelated - np.mean(xy_voxelated, axis=0)
        # Singluar Value decomposition (TODO see if we prefer np.egh)
        _, _, v = np.linalg.svd(xy_centered, full_matrices=False)
        return v[0]
