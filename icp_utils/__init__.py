import numpy as np
from typing import Tuple, Optional
import numpy.typing as npt
from scipy.spatial.transform import Rotation


def icp_2d(points1: np.ndarray, points2: np.ndarray, max_iter: Optional[int] = 1, tol_threshold: Optional[float] = 0.001 ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float, float]:
    '''points1 and points2 should be (n,2)'''
    assert points1.shape[1] == 2
    assert points2.shape[1] == 2

    points1_3d = np.c_[points1, np.zeros(points1.shape[0])]
    points2_3d = np.c_[points2, np.zeros(points2.shape[0])]

    ret_3d = icp_3d(points1_3d, points2_3d, max_iter, tol_threshold)

    #does this work lol
    ret_val = (ret_3d[0][:-1,:-1], ret_3d[1][:-1], ret_3d[2], ret_3d[3])
    return ret_val


def icp_3d(points1: np.ndarray, points2: np.ndarray, max_iter: Optional[int] = 10, tol_threshold: Optional[float] = 0.001 ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float, float]:
    debug_print = True
    '''points1 and points2 should be (n,3)'''
    assert points1.shape[1] == 3
    assert points2.shape[1] == 3
    
    reverse = False

    # remove outliers from both
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    distances1 = np.array([np.linalg.norm(points1[n] - centroid1) for n in range(points1.shape[0])])
    p1_p25, p1_p75 = np.percentile(distances1, [25, 75])
    p1_outlier_threshold = 2.0 * (p1_p75 - p1_p25)
    if debug_print:
        print(f"Outlier threshold for point set 1 is {p1_outlier_threshold}.")
    _p1_temp = []
    for i in range(distances1.shape[0]):
        if distances1[i] <= p1_p75 + p1_outlier_threshold:
            _p1_temp.append(points1[i])
    distances2 = np.array([np.linalg.norm(points2[n] - centroid2) for n in range(points2.shape[0])])
    p2_p25, p2_p75 = np.percentile(distances2, [25, 75])
    p2_outlier_threshold = 2.0 * (p2_p75 - p2_p25)
    if debug_print:
        print(f"Outlier threshold for point set 2 is {p2_outlier_threshold}.")
    _p2_temp = []
    for i in range(distances2.shape[0]):
        if distances2[i] <= p2_p75 + p2_outlier_threshold:
            _p2_temp.append(points2[i])
    
    if len(_p1_temp) > len(_p2_temp):
        _points1 = np.array(_p2_temp).astype(np.float32)
        _points2 = np.array(_p1_temp).astype(np.float32)
        reverse = True
    else:
        _points1 = np.array(_p1_temp).astype(np.float32)
        _points2 = np.array(_p2_temp).astype(np.float32)

    _points1 = 1000.0 * _points1
    _points2 = 1000.0 * _points2

    scale = 1.0
    transform_matrix = np.identity(3)
                                   
    # we're going to throw in the difference between the centroids as an initial guess for the translate vector
    centroid1 = np.mean(_points1, axis=0)
    centroid2 = np.mean(_points2, axis=0)
    translate_vector = centroid2 - centroid1
    if debug_print:
        print(f"Estimated initial translation is {translate_vector}.")


    last_error = None

    for i in range(max_iter):
        # idx of closest point in points2 for each point in points1, and distance to that point
        # this is kind of a jank way of finding closest matches but i'd really like it not to be O(n^2+), it's fine for a proof of concept
        # it wasn't fine for a proof of concept.
        # TODO: improve algorithm. Possibility: just get every point in set 1's distance from every point in set 2, O(n^2) but should be pretty fast for the set sizes we're working with
        # Then sort each of set 1's points' distances, then sort the set of all points on minimum distance for each, then use that as priority for assigning correlations
        # might still have some blind spots but definitely better than possibly letting all the low-indexed points 'hog' all of the good matches
        used_points = set()
        p1_transformed = ( scale * (transform_matrix @ _points1.T).T ) + translate_vector
        all_distances = sorted([ (p1_i, sorted([(p2_i, np.linalg.norm(p1_transformed[p1_i] - _points2[p2_i])) for p2_i in range(_points2.shape[0])], key = lambda p2_t : p2_t[1] ) ) for p1_i in range(p1_transformed.shape[0]) ], key=lambda p1_t: p1_t[1][0][1] )
        # paper that suggested scale-adaptive solution also said best results were found by forcing one-to-one matching on first iter, relaxing it afterwards
        # i'mma force it on at least the first 2 iterations
        p2_correlated = np.zeros_like(p1_transformed)
        for p1_i in all_distances:
            for point_pair in p1_i[1]:
                if point_pair[0] not in used_points:
                    p2_correlated[p1_i[0]] = _points2[point_pair[0]]
                    used_points.add(point_pair[0])
                    break


        centroid1 = np.mean(p1_transformed, axis=0)
        centroid2 = np.mean(p2_correlated, axis=0)

        p1_normalized = p1_transformed - centroid1
        p2_normalized = p2_correlated - centroid2


        #sum_vals = np.array([[np.sum(p1_transformed[:,i] * p2_correlated[:,j]) for j in range(3)] for i in range(3)])

        #rot_matrix_solvable = np.array([ [ (sum_vals[0,0] + sum_vals[1,1] + sum_vals[2,2]), (sum_vals[1,2] - sum_vals[2,1]), (sum_vals[2,0] - sum_vals[0,2]), (sum_vals[0,1] - sum_vals[1,0]) ],
        #                                 [ (sum_vals[1,2] - sum_vals[2,1]), (sum_vals[0,0] - sum_vals[1,1] - sum_vals[2,2]), (sum_vals[0,1] + sum_vals[1,0]), (sum_vals[2,0] + sum_vals[0,2]) ],
        #                                 [ (sum_vals[2,0] - sum_vals[0,2]), (sum_vals[0,1] + sum_vals[1,0]),(-sum_vals[0,0] + sum_vals[1,1] - sum_vals[2,2]), (sum_vals[1,2] + sum_vals[2,1]) ],
        #                                 [ (sum_vals[0,1] - sum_vals[1,0]), (sum_vals[2,0] + sum_vals[0,2]), (sum_vals[1,2] + sum_vals[2,1]),(-sum_vals[0,0] - sum_vals[1,1] + sum_vals[2,2]) ]] )

        # compute cross-covariance matrix of p1 and p2
        # getting an nxn matrix product out of two n-length arrays in numpy is p a i n
        ccov = np.mean(np.array([(p1_normalized[n].reshape(3,1) @ p2_normalized[n].reshape(1,3)) for n in range(p1_normalized.shape[0])]), axis=0)# - (np.mean(p1_transformed, axis=0).reshape(3,1) @ np.mean(_points2, axis=0).reshape(1,3))

        # per https://www.computer.org/csdl/journal/tp/1992/02/i0239/13rRUxEhFtD computing this matrix and getting the eigenvector corresponding to the largest eigenvalue should get you the optimal rotation to try in quaternion form
        # in other words, ✨magic happens here✨
        A_ij = ccov - ccov.T
        cyclic = np.array([A_ij[1,2],A_ij[2,0],A_ij[0,1]])
        Q = np.zeros((4,4))
        Q[0,0] = np.trace(ccov)
        Q[0,1:] = cyclic
        Q[1:,0] = cyclic
        Q[1:,1:] = ccov + ccov.T - (np.trace(ccov) * np.identity(3))
        import pdb; pdb.set_trace()

        eigvals, eigvecs = np.linalg.eig(Q)

        # scipy: Each row is a quaternion representing a rotation in SCALAR-LAST FORMAT WHYYY ARLKAHGDSHGJKHSDFB
        rotation = Rotation.from_quat(np.roll(eigvecs[eigvals.argmax()], -1)).as_matrix()

        # now let's get scale
        # ref https://www.sciencedirect.com/science/article/abs/pii/S1524070321000187?via%3Dihub
        
        p1_temp_rot = (rotation @ p1_normalized.T).T + centroid1
        '''
        import pdb; pdb.set_trace()
        p1_sum = np.sum(p1_temp_rot, axis=0)        
        sol_matrix_a = np.zeros((4,4))
        sol_matrix_a[0,0] = np.sum(np.array([np.dot(p1_temp_rot[n], p1_temp_rot[n]) for n in range(p1_temp_rot.shape[0])]), axis=0)
        sol_matrix_a[0,1:] = p1_sum
        sol_matrix_a[1:,0] = p1_sum
        sol_matrix_a[1,1] = p1_temp_rot.shape[0]
        sol_matrix_a[2,2] = p1_temp_rot.shape[0]
        sol_matrix_a[3,3] = p1_temp_rot.shape[0]

        p2_sum = np.sum(p2_correlated, axis=0)
        sol_matrix_b = np.zeros((4,1))
        sol_matrix_b[0,0] = np.sum([np.dot(p1_temp_rot[n], p2_correlated[n]) for n in range(p1_temp_rot.shape[0])])
        sol_matrix_b[1:,0] = p2_sum

        scale_translate_solution = np.linalg.inv(sol_matrix_a) @ sol_matrix_b
        assert scale_translate_solution.shape[0] == 4
        new_scale = scale_translate_solution[0]
        new_translate = np.squeeze(scale_translate_solution[1:])
        '''
        new_centroid = np.mean(p1_temp_rot, axis=0)
        new_translate = centroid2 - new_centroid


        # get error
        p1_temp_rot = p1_temp_rot + new_translate
        error = np.mean([np.linalg.norm(p1_temp_rot[n] - p2_correlated[n]) for n in range(p1_temp_rot.shape[0])])
        print(f"Error for iteration {i}: {error}")
        if i > 0:
            if error > last_error:
                print(f"Warning: Error increased since last iteration ({last_error})" )

        last_error = error

        transform_matrix = rotation @ transform_matrix # really hope this is the right order
        translate_vector += new_translate
        
        if error <= tol_threshold:
            break

    if reverse:
        transform_matrix = np.linalg.inv(transform_matrix)
        translate_vector = -1 * translate_vector
        scale == 1.0/scale
    
    translate_vector = translate_vector / 1000.0
    
    return transform_matrix, translate_vector, scale, error



        

