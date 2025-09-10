from collections import defaultdict
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

class RadarDiscretizer(object):
    """
    Discretize radar points range, azimuth[0], azimuth[1], vel, RCS into a grid structure
    of discretization_steps,discretization_steps,5 with the first two feature dimensions
    holding the residual of range and azimuth[0]
    """
    def __init__(self, xmin, xmax, ymin, ymax, discretization_steps=128, 
                 valid_indicator=1.0, radius_step=0.5, bufffer_points=0.01,
                 greedy=False):
        self._xmin = xmin
        self._xrange = xmax-xmin
        self._ymin = ymin
        self._yrange = ymax-ymin
        self._discretization_steps = discretization_steps
        self._xstep = self._xrange / discretization_steps
        self._ystep = self._yrange / discretization_steps
        self._valid_indicator = valid_indicator 
        # valid_indicator if None, will not prepend a new chanel dimension for the valid indicator

        self._buffer_points = bufffer_points
        self.radius_step = radius_step
        self.greedy = greedy # whether to preassign the closest point to the centre in a contested cell
        """self.neighborhood_indices[i] contains the pairs (x,y) such that
           sqrt(x^2 + y^2) is in (i*self.radius_step, (i+1)*self.radius_step]
        """
        self.neighborhood_indices = [ np.zeros((0, 2), dtype=int) ] if greedy else [ np.zeros((1, 2), dtype=int) ]
        # (0,0) is not a valid index for a neighborhood with the current logic
        self.prepare_neighborhood_indices(10)

    @property
    def xmin(self):
        return self._xmin
    @property
    def xrange(self):
        return self._xrange
    @property
    def xstep(self):
        return self._xstep
    @property
    def xmax(self):
        return self._xmin+self._xrange
    @xmax.setter
    def xmax(self, value):
        self._xrange = value - self._xmin
        self._xstep = self._xrange / self._discretization_steps

    @property
    def ymin(self):
        return self._ymin
    @property
    def yrange(self):
        return self._yrange
    @property
    def ystep(self):
        return self._ystep
    @property
    def ymax(self):
        return self._ymin+self._yrange
    @ymax.setter
    def ymax(self, value):
        self._yrange = value - self._ymin
        self._ystep = self._yrange / self._discretization_steps

    @property
    def discretization_steps(self):
        return self._discretization_steps

    @property
    def valid_indicator(self):
        return self._valid_indicator

    @property
    def largest_prepared_radius(self):
        """
        return the largest radius for which neighborhood indices are prepared
        """
        return len(self.neighborhood_indices) * self.radius_step

    def prepare_neighborhood_indices(self, radius):
        """
        Precompute the neighborhood indices for a given radius (and less)
        """
        min_idx = len(self.neighborhood_indices)
        max_idx = int((radius - 1e-4) // self.radius_step)+1
        existing_radius = min_idx * self.radius_step
        max_radius = int(radius+1)
        #print(f"Preparing neighborhood indices for radius {radius} ({min_idx*self.radius_step:.2f}, {(max_idx)*self.radius_step:.2f}] e.g. index range [{min_idx} to {max_idx})")
        if existing_radius < radius:
            brackets = defaultdict(list)
            for x in range(0, max_radius):
                for y in range(x, max_radius):
                    dist = int((np.sqrt(x**2 + y**2) - 1e-4) // self.radius_step)
                    #print(f"[{x}, {y}] -> {np.sqrt(x**2 + y**2):.2f} in ({dist*self.radius_step:.2f}, {dist*self.radius_step+self.radius_step:.2f}] -> {dist} ")
                    if min_idx <= dist and dist < max_idx:
                        brackets[dist].append((x, y))
            for i in range(min_idx, max_idx):
                pairs = brackets[i]
                #print(f"{i}: Pairs for radius ({i*self.radius_step:.1f}, {(i+1)*self.radius_step:.1f}]: {pairs}")
                indices = np.empty((8*len(pairs), 2), dtype=int)
                cumidx = 0
                for x, y in pairs:
                    if x == 0:
                        if y == 0:
                            indices[cumidx] = [0, 0]
                            cumidx += 1
                        else:
                            indices[cumidx]     = [ 0,  y]
                            indices[cumidx + 1] = [ 0, -y]
                            indices[cumidx + 2] = [ y,  0]
                            indices[cumidx + 3] = [-y,  0]
                            cumidx += 4
                    elif x == y:
                        indices[cumidx]     = [ x,  y]
                        indices[cumidx + 1] = [ x, -y]
                        indices[cumidx + 2] = [-x,  y]
                        indices[cumidx + 3] = [-x, -y]
                        cumidx += 4
                    else: #if x != y:
                        indices[cumidx]     = [ x,  y]
                        indices[cumidx + 1] = [ x, -y]
                        indices[cumidx + 2] = [-x,  y]
                        indices[cumidx + 3] = [-x, -y]
                        indices[cumidx + 4] = [ y,  x]
                        indices[cumidx + 5] = [ y, -x]
                        indices[cumidx + 6] = [-y,  x]
                        indices[cumidx + 7] = [-y, -x]
                        cumidx += 8
                assert len(self.neighborhood_indices) == i, f"len(self.neighborhood_indices) {len(self.neighborhood_indices)} != {i}"
                self.neighborhood_indices.append(indices[:cumidx])

    def scale_xy(self, x, y):
        return (x - self.xmin) / self.xstep, (y - self.ymin) / self.ystep

    def unscale_xy(self, x_scaled, y_scaled):
        return x_scaled * self.xstep + self.xmin, y_scaled * self.ystep + self.ymin

    def xy_to_rowcol(self, x, y):
        x_scaled, y_scaled = self.scale_xy(x, y)
        row = np.clip(int(x_scaled), 0, self.discretization_steps - 1)
        col = np.clip(int(y_scaled), 0, self.discretization_steps - 1)
        return row, col

    def data_to_gridentry(self, x_off, y_off, features):
        return np.concatenate((
            [self.valid_indicator] if self.valid_indicator is not None else [], 
            [x_off, y_off],
            features
        ))
    
    def point_to_gridentry(self, point, row, col):
        """ map a point to a grid entry """ 
        scaled_x, scaled_y = self.scale_xy(*point[:2])
        # without the 0.5, the residuls are between 0 and 1, 
        # but we'de like to have them centered on 0
        return self.data_to_gridentry(
            x_off=scaled_x - (row + 0.5),
            y_off=scaled_y - (col + 0.5),
            features=point[2:]
        )
        
    def point_from_gridentry(self, grid, row, col):
        """ this function inverts the action of point_to_gridentry """
        entry = grid[row, col, :]
        x, y = self.unscale_xy(entry[1] + (row + 0.5), entry[2] + (col + 0.5))
        return np.block([x, y, entry[3:]])
    
    def to_grid(self, points, grid=None):
        """
        Map RSPClusters x, y, azimuth_1, vel, RCS to grid according to x, y position
        grid has dimensions (self.discretization_steps, self.discretization_staps, 6)
        with the first channel indicating a valid entry if set to self.mask_value
        """
        #start_time = time.process_time()
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        feature_dim = points.shape[-1]
        grid_shape = (
            self.discretization_steps, 
            self.discretization_steps, 
            feature_dim + (0 if self.valid_indicator is None else 1) )
        if grid is None:
            grid = np.zeros(grid_shape, dtype=np.float32)
        else:
            assert grid.shape == grid_shape
        occupied_grid = np.zeros( # initiated with False everywhere
            (self.discretization_steps, self.discretization_steps), dtype=bool
        )

        # Map points to grid cells and back
        # there may be multiple points mapped to the same grid cell
        # but a single point is mapped is mapped to only one cell
        
        # Vectorized computation of grid cells for all points
        scaled_x, scaled_y = self.scale_xy(points[:, 0], points[:, 1])
        rows = np.clip(scaled_x.astype(int), 0, self.discretization_steps - 1)
        cols = np.clip(scaled_y.astype(int), 0, self.discretization_steps - 1)

        # For each point, find the corresponding grid cell
        # (each cell has a unique id given by its row and column coordinates)
        cell_ids = rows * self.discretization_steps + cols
        # for each cell id, find the points that belong to it
        index_sorted_cell_ids = np.argsort(cell_ids, kind="stable")
        filled_cells, same_cell_start = np.unique(
            cell_ids[index_sorted_cell_ids],
            return_index=True)
        


        # Process each filled cell
        remaining_points = {}
        nbr_remaining_points = 0
        for i in range(len(filled_cells)):
            # Get the row and column for this cell
            cell_id = filled_cells[i]
            row = cell_id // self.discretization_steps
            col = cell_id % self.discretization_steps
            
            # Get the point indices that map to this cell
            start_idx = same_cell_start[i]
            try:
                end_idx = same_cell_start[i+1]
            except IndexError:
                end_idx = len(index_sorted_cell_ids)
            point_indices = index_sorted_cell_ids[start_idx:end_idx]
            
            # If only one point maps to this cell, directly assign it
            if self.greedy:
                if len(point_indices) == 1:
                    idx = point_indices[0]
                    grid[row, col, :] = self.data_to_gridentry(
                        x_off=scaled_x[idx] - (row + 0.5),
                        y_off=scaled_y[idx] - (col + 0.5),
                        features=points[idx, 2:],
                    )
                    occupied_grid[row, col] = True
                else:
                    # For cells with multiple points, select the one closest to the cell center
                    x_cell_offset = scaled_x[point_indices] - (row + 0.5)
                    y_cell_offset = scaled_y[point_indices] - (col + 0.5)
                    distance_from_cell_center = x_cell_offset**2 + y_cell_offset**2
                    list_idx = np.argmin(distance_from_cell_center)
                    
                    grid[row, col, :] = self.data_to_gridentry(
                        x_off=x_cell_offset[list_idx],
                        y_off=y_cell_offset[list_idx],
                        features=points[point_indices[list_idx], 2:],
                    )
                    occupied_grid[row, col] = True
                        
                    # Track remaining points that weren't assigned to their preferred cell
                    mask = np.ones(len(point_indices), dtype=bool)
                    mask[list_idx] = False
                    remaining_points[(row,col)] = point_indices[mask]
                    nbr_remaining_points += len(point_indices) - 1
            else:
                # For cells with multiple points, store the point indices for later processing
                remaining_points[(row,col)] = point_indices
                nbr_remaining_points += len(point_indices)

        #after_cell_fill_time = time.process_time()

        contested_cells = np.array(list(remaining_points.keys()))
        candidate_cells = []
        nbr_candidate_cells = 0
        radius_idx = 0
        buffer_points = int(max(0, self._buffer_points * (self.discretization_steps**2 - 2*nbr_remaining_points)))
        non_buffer_points = None
        # TODO: Simple solution if there are more radar points than grid cells
        while nbr_candidate_cells < nbr_remaining_points + buffer_points:
            if len(self.neighborhood_indices) <= radius_idx:
                self.prepare_neighborhood_indices(radius_idx * self.radius_step)

            neighborhood_indices = self.neighborhood_indices[radius_idx]
            if len(neighborhood_indices) == 0:
                radius_idx += 1
                continue

            pos_rows = (contested_cells[:, 0][:, None] +
                        neighborhood_indices[:, 0][None,:]).ravel()
            pos_cols = (contested_cells[:, 1][:, None] +
                        neighborhood_indices[:, 1][None,:]).ravel()
            valid_mask = np.logical_and.reduce([
                pos_rows >= 0, 
                pos_rows < self.discretization_steps,
                pos_cols >= 0, 
                pos_cols < self.discretization_steps])
            unique_pos_cells = np.unique(np.column_stack((pos_rows[valid_mask], pos_cols[valid_mask])), axis=0)
            new_cells = np.where(occupied_grid[unique_pos_cells[:, 0], unique_pos_cells[:, 1]] == False)[0]
            nbr_candidate_cells += len(new_cells)
            occupied_grid[unique_pos_cells[new_cells, 0], unique_pos_cells[new_cells, 1]] = True
            candidate_cells.append(unique_pos_cells[new_cells])
            radius_idx += 1
            if non_buffer_points is None and nbr_candidate_cells >= nbr_remaining_points:
                non_buffer_points = nbr_candidate_cells

        #after_neighborhood_time = time.process_time()

        candidate_cells = np.concatenate(candidate_cells, axis=0)
        remaining_point_indices = np.concatenate(list(remaining_points.values()), axis=0)
        remaining_points = np.column_stack((scaled_x[remaining_point_indices],
                                            scaled_y[remaining_point_indices]))
        #print(f"Found {nbr_candidate_cells} ({candidate_cells.shape}) candidate cells, "+
        #      f"searching in radius {radius_idx*self.radius_step} around {nbr_remaining_points} "+
        #      f"({remaining_points.shape}) remaining points in {len(contested_cells)} "+
        #      f"contested cells. Without buffer of {buffer_points} points, "+
        #      f"we would have {non_buffer_points} candidate cells.")
        #print(f"remaining points:\n{remaining_points[:10]}\n...\ncandidate_cells:\n{candidate_cells[:10]}\n...\n")

        # This is the actually most expensive part of the algorithm
        cost_matrix = (np.sum( (remaining_points[:,None,:] - candidate_cells[None,:,:] -0.5)**2, axis=2))

        #pre_linear_sum_assignment_time = time.process_time()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        #post_linear_sum_assignment_time = time.process_time()

        for i, j in zip(row_ind, col_ind):
            point_idx = remaining_point_indices[i]
            row, col = candidate_cells[j]
            #print(f"Assigning point {point_idx} ({scaled_x[point_idx]}, {scaled_y[point_idx]}) to cell {(row, col)}")
            grid[row, col, :] = self.data_to_gridentry(
                x_off=scaled_x[point_idx] - (row + 0.5),
                y_off=scaled_y[point_idx] - (col + 0.5),
                features=points[point_idx, 2:],
            )
        
        #grid_filling_time = time.process_time()

        # Time logging:
        #print(f"Needed time: {grid_filling_time - start_time:.6f} - "+
        #      f"for cell filling: {after_cell_fill_time - start_time:.6f} - "+
        #      f"for neighborhood search: {after_neighborhood_time - after_cell_fill_time:.6f} - "+
        #      f"cost matrix creation: {pre_linear_sum_assignment_time - after_neighborhood_time:.6f} - "+
        #      f"for linear sum assignment: {post_linear_sum_assignment_time - pre_linear_sum_assignment_time:.6f} - "+
        #      f"for grid filling: {grid_filling_time - post_linear_sum_assignment_time:.6f}")

        # Reconstruction loss. If not close to zero (e.g. 1e-12) something went wrong
        #print(f"maximal distance between reconstructed point and closest original point:",
        #      f"{np.max(np.min(np.sum((self.to_points(grid)[:, None, :] - points[None, :, :])**2, axis=2), axis=1), axis=0):.2e}")
        #print()
        return grid

    def to_points(self, grid):
        """
        takes an input array of size discretization_steps, discretization_steps, feature_dim
        and returns a list of points from 
        """
        if self.valid_indicator is not None:
            row_idx, col_idx = np.where(grid[..., 0] == self.valid_indicator)
        else:
            row_idx, col_idx = np.where(grid[..., 0] > 1e-4)

        return np.array([
            self.point_from_gridentry(grid, r, c)
            for (r, c) in zip(row_idx, col_idx)
        ])

    def grid_to_image(self, grid, swap_xy=False, invert_rows=False, invert_columns=False):
        """
        Radar grids have x, y in rows and columns, respectively.
        to display them as images alonside scatter plots, one may need to swap x and y
        and invert the order of columns and rows after swapping
        """
        grid = np.swapaxes(grid, 0, 1) if swap_xy else grid
        row_step = -1 if invert_rows else 1
        col_step = -1 if invert_columns else 1
        return grid[::row_step, ::col_step, :]
