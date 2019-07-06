"""
Semi-supervised k-means
Author: Bin Feng
Date: 06/10/2019
"""

from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from collections import Counter
# reference to
# https://github.com/lsxliron/SemiSupervisedKMeans
# https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/cluster/k_means_.py#L769

class SemiKMeans(object):
    """
    Attributes:
        n_clusters: int, default: 8
            the number of clusters to form as the number of centroids to generate
        ### init: str, {'k-means++', 'random'}, default: k-means++
            the method used to initialized centers
            "k-means++": seelects initial centers in a smart way to speed up conveergence
            "random": chooses k observations (rows) at random as initialized centers ###
        max_iter: int, default: 300
            max number of iterations of the semi-supervised k-means to run
        distance_metric: str, {"euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski"} default: euclidean
            metric used for pairwise distance calculation
        labeled_data: list, 2D array
            every cluster should be included in this 2D array including clusters with none labeled data. For example, 
            if we know some labeled data for cluster 0 and 2, and the total number of clusters is 3. Then, we have
            ```
            labeled_data = [np.array([1,2]), np.array([]), np.array([3,4])]
            labeled_data with no element is not expected: labeled_data = [np.array([])]
            ```
        weight: float, default: 0.5
            given different weight for labeled_data, range from [0, +inf)
        verbose: boolen, default: 0
            prints iterations and convergence rate if is 1
    """
    
    def __init__(self, n_clusters=8, max_iter=300, distance_metric='euclidean', labeled_data=None, 
                 weight=0.5, tol=0.0001, verbose=0):
        """initialize SemiKmeans object"""
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.labeled_data = labeled_data
        self.weight = float(weight)
        self.tol = tol
        self.verbose = verbose
        
        self.centroids = None
        self.label_ = None
        
    
    def _get_distance(self, x, y):
        """ 
            :param x: np.array, point 1
            :param y: np.array, point 2
            :return: float, distance between point 1 and point 2 defined by metric
        """
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        
        return pairwise_distances(x, y, metric = self.distance_metric)[0]
    
    def _update_centroids(self, data):
        """
            updating centroid by taking average of all points in a cluster
            :param data: np.array all points coordinates
        """
        for i in range(self.n_clusters):
            self.centroids[i] = np.mean(data[np.where(self.label_ == i)], axis = 0)
    
    def _update_biased_centroids(self, data):
        """
            updating centroid for semi-supuervised learning
            :param data: np.array all points coordinates
        """
        weights = np.ones(data.shape[0])   
        labeled_data_idx = np.hstack(np.array(self.labeled_data).flat).astype(np.int)
        weights[labeled_data_idx] = self.weight
        weights = weights/weights.sum()   
        
        for i in range(self.n_clusters):
            # compute centroid for every cluster
            inds = np.where(self.label_==i)[0]
            self.centroids[i] = np.average(data[inds], weights=weights[inds], axis=0)
        
        ''' if labeled_data_idx needed to be updated
        for i in xrange(self.n_clusters):
            if i<len(self.known_data) and len(self.known_data[i]):
                max_vote = 0
                max_vote = map(lambda lbl: (self.label_[self.known_data[i]]==lbl).sum() , range(0, self.n_clusters))
                self.label_[self.known_data[i]] = np.argmax(max_vote)
        '''
    
    def _kmeans_pp(self, data, n_local_trials = None):
        """
            Initialize cluster centers using k-means++
            :param data: np.array all points coordinates
            :param n_local_trials : integer, optional
                The number of seeding trials for each center (except the first/from labeled data),
                of which the one reducing inertia the most is greedily chosen. Set to None to make 
                the number of trials depend logarithmically on the number of seeds (2+log(k)); this 
                is the default.
            
            Note
            -----
            Selects initial cluster centers for k-mean clustering in a smart way
            to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
            "k-means++: the advantages of careful seeding". ACM-SIAM symposium
            on Discrete algorithms. 2007
        """
        # initialize labels to be the same size as data
        self.label_ = np.zeros(len(data))
        self.label_.fill(-1)
        
        # If we have some labeled data points, make them the centroid
        # otherwise, follow k-means++ procedures
        if self.labeled_data is not None:
            current_centers = None
            for i, pts in enumerate(self.labeled_data):
                if len(pts):
                    self.label_[pts] = i
                    if current_centers is not None:
                        current_centers = np.vstack((current_centers, np.mean(data[np.where(self.label_==i)], axis=0)))
                    else:
                        current_centers = np.mean(data[np.where(self.label_==i)], axis=0)
                        current_centers = current_centers.reshape(1,len(current_centers))
        
        else:
            # Choose the first centroid randomly as stated in kmeans++ procedures
            first_centroid_index = np.random.choice(np.arange(0, len(data)), 1)
            self.label_[first_centroid_index] = 0
            current_centers = data[first_centroid_index,]
        
        # initialize the rest centroids
        for i in range(len(current_centers), self.n_clusters):
            found_centroid = False
            distances = np.array([min([self._get_distance(center, p) for center in current_centers]) for p in data], dtype=np.float64)
            distances_sq = distances**2
            probabilities = distances_sq/distances_sq.sum()
            
            # Set the number of local seeding trials.
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(self.n_clusters))
            
            rand_vals = np.random.random_sample(n_local_trials)
            candidate_ids = np.searchsorted(np.cumsum(probabilities), rand_vals)
            
            # Decide which candidate is the best
            best_candidate = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # Compute potential when including center candidate
                new_dist_sq = distances_sq[candidate_ids[trial]]

                # Store result if it is the best local trial so far
                if (best_candidate is None) or (new_dist_sq < best_dist_sq):
                    best_candidate = candidate_ids[trial]
                    best_dist_sq = new_dist_sq

            current_centers = np.vstack((current_centers, data[best_candidate]))
            
            next_label = 0
            while next_label in set(self.label_):
                next_label += 1
            
            self.label_[best_candidate] = next_label
            
        self.centroids = current_centers

        
    def predict(self):
        """
            :return: np.array, labels of the clustered data
        """
        return self.label_
        
    def fit_predict(self, data):
        """
            Clusteres the data and returns tha labels
            :param data: np.ndarray, data to cluster
            :return: np.array, data labels
        """
        self.fit(data)
        return self.predict()

    def fit(self, data):
        """
            Clusters the data
            :param data: np.ndarray, data to cluster
        """
        # initializa centroids, labels, iteration times, threshold
        self._kmeans_pp(data)
        new_label = self.label_.copy()
        counter = 0
        tol = np.infty
        labeled_data_idx = np.hstack(np.array(self.labeled_data).flat).astype(np.int)
        while counter < self.max_iter and self.tol < tol:
            for i, p in enumerate(data):
                if i not in labeled_data_idx:
                    new_label[i] = np.argmin([self._get_distance(center, p) for center in self.centroids])
            
            
            print(Counter(new_label))
            old_centroids = self.centroids.copy()
            self.label_ = new_label.copy()
            
            if self.labeled_data is not None:
                self._update_biased_centroids(data)
            else:
                self._update_centroids(data)
                
            tol = abs(np.mean(old_centroids)-np.mean(self.centroids))

            counter+=1
            
            if self.verbose:
                print("Iteration {}\tConvergance: {}".format(counter, tol))
