import numpy as np
from sklearn.decomposition import PCA
import os
import graphviz

from ..data_utils.processing import time_delay_embed, partial_corr, corr

# Cross Mapping (CM) and Partial CM for Causal Inference
# based on simplex projection between two embeddings 
# (inputs are embeddings not necessarily of the same dimensions)
# Insprired by: https://github.com/PrinceJavier/causal_ccm/blob/main/causal_ccm/causal_ccm.py


# 1. Cross Mapping
class CM_simplex:
    """ Cross mapping based on simplex projection (kNN) between two embeddings (inputs are embeddings not necessarily of the same dimensions)
        
    Inputs: 
        df: dataframe containing multivariate time series data, each column as a variable to be accessed by column name
        causes: list of namestrings of cause variables;
        effects: list of namestrings of effect variables;
        tau: time delay for time delay embedding;
        emd: embedding dimension for time delay embedding;

        knn (int): number of nearest neighbors to use for the simplex projection (default: 10

        L (int): the length of the time series to use (default: 1000)


        the first dimensions of M_cause and M_effect should be the same, representing the time indices;
        second dimensions can be different, representing the embedding dimensions.

    """

    def __init__(self, df, causes, effects, tau=2, emd=8, knn=10, L=3000, method='vanilla',**kwargs):
        self.df = df
        self.causes = causes
        self.effects = effects

        self.tau = tau
        self.emd = emd
        
        self.M_cause=CM_simplex._time_delay_embed(df[causes], tau, emd, L)
        self.M_effect=CM_simplex._time_delay_embed(df[effects], tau, emd, L)

        self.knn = knn

        self.method = method # 'vanilla' or 'PCA'

        self.kwargs = kwargs # dictionary of other parameters (PCA dims)

        assert self.M_cause.shape[0] == self.M_effect.shape[0], "The first dimensions of M_cause and M_effect should be the same, representing the time indices."

        self.model=CM_rep_simplex(cause_reps=self.M_cause, effect_reps=self.M_effect, knn=knn, L=L, method=method, **kwargs)

    def predict_manifolds(self):
        """ Cross Mapping Prediction:
        Reconstruct the manifolds of cause and effect.
        use kNN weighted average to get the reconstruction of the two manifolds, 
        return the two reconstructions.
        """
        return self.model.predict_manifolds()



    def causality(self):
        """ Causality score (error and averaged pearson correlation)"""
        return self.model.causality()

    @staticmethod
    def _time_delay_embed(df, tau, emd, L):
        """ Process the input dataframe to time delay embedding.
        Need to process each univariate time series one by one, then stack together.
        """
        embed = []
        for col in df.columns:
            ts = df[col].values
            embed.append(time_delay_embed(ts, tau, emd, L))
        embed = np.concatenate(embed, axis=1)
        return embed

    


    
# 2. Partial Cross Mapping
class PCM_simplex(CM_simplex):
    """ Partial Cross Mapping based on simplex projection (kNN) between two embeddings (inputs are embeddings not necessarily of the same dimensions)
        
    Inputs: 
        df: dataframe containing multivariate time series data, each column as a variable to be accessed by column name
        causes: list of namestrings of cause variables;
        effects: list of namestrings of effect variables;
        cond: list of namestrings of conditioning variables;

        tau: time delay for time delay embedding;
        emd: embedding dimension for time delay embedding;

        knn (int): number of nearest neighbors to use for the simplex projection (default: 10

        L (int): the length of the time series to use (default: 1000)


        the first dimensions of M_cause and M_effect should be the same, representing the time indices;
        second dimensions can be different, representing the embedding dimensions.

    
    -----------------------------

        The prediction procedure is modified: 
        1. Need 3 inputs: 
            cause, effect, condition

        2. According to the PCM paper, suppose X1->X_cond->X2,
            we don't know if there is direct causation between X1 and X2 only by CCM.
            Suppose cause = X1, effect = X2, condition = Xcond.
            
            To obtain: 
            "M_cause_reconst1" - the CM estimate of cause from effect;
            (first, "M_cond_reconst" - the CM estimate of M_cond from effect);
            then, "M_cause_reconst2" - the CM estimate of cause from "M_cond_reconst.

        3. Compute the partial correlation: ParCorr(X1, X1_reconst1 | X1_reconst2):
            Intuition is that now the information flow through the intermediate Xcond is eliminated,
            so if there is still a strong correlation between X1 and X1_reconst1,
            then X1 and X2 are directly causally related.
            The larger the ParCorr, the stronger the direct causation.


    """

    def __init__(self, df, causes, effects, cond, tau=2, emd=8, knn=10, L=3000, method='vanilla', **kwargs):
        super().__init__(df, causes, effects, tau, emd, knn, L, method, **kwargs)
        self.cond = cond
        self.M_cond = super()._time_delay_embed(df[cond], tau, emd, L)

        assert self.M_cause.shape[0] == self.M_effect.shape[0] == self.M_cond.shape[0], "The first dimensions of M_cause, M_effect, and M_cond should be the same, representing the time indices."

        self.model=PCM_rep_simplex(cause_reps=self.M_cause, effect_reps=self.M_effect, cond_reps=self.M_cond, knn=knn, L=L, method=method, **kwargs)

    def predict_manifolds(self):
        """ Partial Cross Mapping Prediction:
        Overriding the predict_manifolds() method in CM_simplex class.

        use kNN weighted average for reconstruction, 
        return the two reconstructions.
        """
        return self.model.predict_manifolds()
    
    def causality(self):
        """ Causality scores:
        Correlation based:
            1. direct correlation between M_cause and M_cause_reconst1;
            2. partial correlation between M_cause and M_cause_reconst1 given M_cause_reconst2.
            3. the ratio of ParCorr over DirectCorr.
            
        Error based:
            1. direct error between M_cause and M_cause_reconst1;
            2. indirect error between M_cause and M_cause_reconst2.
            3. the ratio of IndirectError over DirectError.
        """

        return self.model.causality()
    


# Utility 1: CM mapping between representations - either the delay embeddings or latent representations
class CM_rep_simplex:
    """ Cross mapping based on simplex projection (kNN) between two representations (inputs are not necessarily of the same dimensions)
        
    Inputs: 
        cause_reps: representation of cause variable;
        effect_reps: representation of effect variable;
        knn (int): number of nearest neighbors to use for the simplex projection (default: 10)

        method (str): the method to use for kNN search, either 'PCA' or 'vanilla' 

    """
    # mise a jour mardi 5 mars 2024 (reimplemented with PCA) - allow another input to determine whether to use PCA or vanilla kNN
    def __init__(self, cause_reps, effect_reps, knn=10, L=None, method='vanilla', **kwargs):
        self.M_cause = cause_reps[:]
        self.M_effect = effect_reps
        self.knn = knn
        self.L = L
        if L is not None:
            self.M_cause = self.M_cause[:L]
            self.M_effect = self.M_effect[:L]
        
        self.method = method
        self.kwargs = kwargs # dictionary of other parameters (PCA dims)
        
        assert self.method=='PCA' or self.method=='vanilla', "The method should be either 'PCA' or 'vanilla'."


        assert self.M_cause.shape[0] == self.M_effect.shape[0], "The first dimensions of cause_reps and effect_reps should be the same, representing the time indices."

    def predict_manifolds(self):
        """ Cross Mapping Prediction:  (mise a jour mardi 5 mars 2024 avec PCA)
        Reconstruct the manifolds of cause and effect.
        use kNN weighted average to get the reconstruction of the two manifolds, 
        return the two reconstructions.
        """
        self.M_cause_reconst=np.zeros(self.M_cause.shape)
        self.M_effect_reconst=np.zeros(self.M_effect.shape)
        
        if self.method=='vanilla':
            self.dists_cause=self.get_distance_vanilla(self.M_cause)
            self.dists_effect=self.get_distance_vanilla(self.M_effect)

            for t_tar in range(self.M_cause.shape[0]):
                # -------The cause manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cause_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)


                # -------The effect manifold reconstruction from the cause -------
                # get the nearest distances of the target point t_tar on the cause manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_cause, t_tar, self.knn)
                # get the weights of the nearest neighbors on the effect manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the cause manifold
                self.M_effect_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_effect[nearest_time_indices], axis=0)

            return self.M_cause_reconst, self.M_effect_reconst

        elif self.method=='PCA':
            # use PCA to reduce the dimension of the representations
            n_comp=self.kwargs['pca_dim'] # PCA component

            # use PCA to reduce the dimension of the representations
            pca_cause=PCA(n_components=n_comp)
            pca_effect=PCA(n_components=n_comp)
            M_cause_pca=pca_cause.fit_transform(self.M_cause)
            M_effect_pca=pca_effect.fit_transform(self.M_effect)

            self.dists_cause=self.get_distance_vanilla(M_cause_pca)
            self.dists_effect=self.get_distance_vanilla(M_effect_pca)

            for t_tar in range(self.M_cause.shape[0]):
                # -------The cause manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cause_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)

                # -------The effect manifold reconstruction from the cause -------
                # get the nearest distances of the target point t_tar on the cause manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_cause, t_tar, self.knn)
                # get the weights of the nearest neighbors on the effect manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the cause manifold
                self.M_effect_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_effect[nearest_time_indices], axis=0)

            return self.M_cause_reconst, self.M_effect_reconst





    


    def causality(self):
        """ vanilla kNN! 
        Causality score (error and averaged pearson correlation)"""
        # if the cause manifold reconstruction from effect is good, the cause->effect relationship is strong;
        # if the effect manifold reconstruction from cause is good, the effect->cause relationship is strong.
        

        M_cause_reconst, M_effect_reconst=self.predict_manifolds()

        # get the causality score (error)
        sc1_error=np.mean(np.sqrt(np.sum((self.M_cause-M_cause_reconst)**2, axis=1)))
        sc2_error=np.mean(np.sqrt(np.sum((self.M_effect-M_effect_reconst)**2, axis=1)))
        
        # get the causality score (pearson correlation) - average over each emd dimension
        sc1_corr=np.nanmean(np.abs(corr(self.M_cause, M_cause_reconst)))
        sc2_corr=np.nanmean(np.abs(corr(self.M_effect, M_effect_reconst)))

        # get the causality score (R2)
        sc1_r2=1-np.sum((self.M_cause-M_cause_reconst)**2)/np.sum((self.M_cause-np.mean(self.M_cause))**2)
        sc2_r2=1-np.sum((self.M_effect-M_effect_reconst)**2)/np.sum((self.M_effect-np.mean(self.M_effect))**2)

        return sc1_error, sc2_error, sc1_corr, sc2_corr, sc1_r2, sc2_r2
    
        


    @staticmethod
    def get_nearest_distances(distM, t_tar, knn=10):
        """ used for vanilla kNN!
        Get the nearest distances of the target point t_tar in distM.

        Input: (2D array)
            distM: Matrix of distances between each pair of points in M, (T_indices x T_indices) array
            t_tar: target time index
            knn: number of nearest neighbors to use for the simplex projection (default: 10)

        Output: (1D array)
            nearest_time_indices: time indices of the nearest neighbors
            nearest_distances: distances of the nearest neighbors in the same order
        """

        # get the distances of the target point t_tar to all other points
        dists=distM[t_tar]

        # get the nearest distances of the target point t_tar
        nearest_time_indices=np.argsort(dists)[1:knn+1]
        nearest_distances=dists[nearest_time_indices]

        return nearest_time_indices, nearest_distances

    

    # @staticmethod
    # def get_distance_vanilla(M):
    #     """ used for vanilla kNN!
    #     Calculate the distances between each pair of points in M.
        
    #     Input: (2D array)
    #         M: embedding of a variable, 2D array of shape (T_indices, embedding_dim)
            
    #     Output: (2D array)
    #         t_steps: time indices
    #         dists: distances between each pair of points in M, (T_indices x T_indices) array
    #     """

    #     # extract the temporal indices
    #     T_max=M.shape[0]

    #     # get the distances between each pair of points in M
    #     dists=np.zeros((T_max,T_max))

    #     # for i in range(T_max):
    #     #     for j in range(T_max):
    #     #         dists[i,j]=np.linalg.norm(M[i]-M[j])

    #     # more efficient loop - only compute half of the matrix, all the diagonal elements are 0
    #     for i in range(T_max):
    #         for j in range(i+1,T_max):
    #             dists[i,j]=np.linalg.norm(M[i]-M[j])
    #             dists[j,i]=dists[i,j]       

    #     return dists
    

    @staticmethod
    def get_distance_vanilla(M, tol=1e-8):
        """ used for vanilla kNN!
        Calculate the distances between each pair of points in M. This will be a faster implementation than the previous one with loops. 
        Credit to: @shubhamrajeevpunekar https://github.com/shubhamrajeevpunekar
        
        Input: (2D array)
            M: embedding of a variable, 2D array of shape (T_indices, embedding_dim)
            
        Output: (2D array)
            dists: distances between each pair of points in M, (T_indices x T_indices) array
        """

        squared_norms=np.sum(np.square(M), axis=1, keepdims=True)
        dot_product=M@M.T
        pairwise_squared_dist=squared_norms+squared_norms.T-2*dot_product

        # numerical stability
        pairwise_squared_dist=np.maximum(pairwise_squared_dist, 0.0) # set the negative values to 0
        pairwise_squared_dist[np.abs(pairwise_squared_dist)<tol]=0.0 # if the distance is very small, set it to 0

        dists=np.sqrt(pairwise_squared_dist)


        return dists

        

    


# Utility 2: PCM mapping between representations - either the delay embeddings or latent representations
class PCM_rep_simplex(CM_rep_simplex):
    """ Partial Cross Mapping based on simplex projection (kNN) between two representations (inputs are not necessarily of the same dimensions)
        
    Inputs: 
        cause_reps: representation of cause variable;
        effect_reps: representation of effect variable;
        cond_reps: representation of conditioning variable;

        knn (int): number of nearest neighbors to use for the simplex projection (default: 10

    """
    def __init__(self, cause_reps, effect_reps, cond_reps, knn=10, L=None, method='vanilla',**kwargs):
        super().__init__(cause_reps, effect_reps, knn, L, method, **kwargs)
        self.M_cond = cond_reps
        if L is not None:
            self.M_cond = self.M_cond[:L]

        assert self.M_cause.shape[0] == self.M_effect.shape[0] == self.M_cond.shape[0], "The first dimensions of M_cause, M_effect, and M_cond should be the same, representing the time indices."

    def predict_manifolds(self):
        """ Partial Cross Mapping Prediction:
        Overriding the predict_manifolds() method in CM_simplex class.

        use kNN weighted average for reconstruction, 
        return the two reconstructions.
        """
        self.M_cause_reconst1=np.zeros(self.M_cause.shape) # direct reconstruction of cause from effect
        self.M_cause_reconst2=np.zeros(self.M_cause.shape) # indirect reconstruction of cause from M_cond_reconst
        self.M_cond_reconst=np.zeros(self.M_cond.shape)
        
        if self.method=='vanilla':
            self.dists_cause=self.get_distance_vanilla(self.M_cause)
            self.dists_effect=self.get_distance_vanilla(self.M_effect)
            self.dists_cond=self.get_distance_vanilla(self.M_cond)
            
            # starting from the effect, first map to reconstruct the condition and directly the cause
            for t_tar in range(self.M_effect.shape[0]):
                # -------The condition manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the condition manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cond_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_cond[nearest_time_indices], axis=0)

                # -------The cause manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cause_reconst1[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)    

            self.dists_cond_reconst=self.get_distance_vanilla(self.M_cond_reconst)

            # starting from the reconstructed condition, map to reconstruct the cause
            for t_tar in range(self.M_cond.shape[0]):
                # -------The cause manifold reconstruction from the reconstructed condition -------
                # get the nearest distances of the target point t_tar on the reconstructed condition manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_cond_reconst, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the reconstructed condition manifold
                self.M_cause_reconst2[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)

            return self.M_cause_reconst1, self.M_cause_reconst2    

        elif self.method=='PCA':
            # use PCA to reduce the dimension of the representations
            n_comp=self.kwargs['pca_dim']
            
            # use PCA to reduce the dimension of the representations
            pca_cause=PCA(n_components=n_comp)
            pca_effect=PCA(n_components=n_comp)
            pca_cond=PCA(n_components=n_comp)
            M_cause_pca=pca_cause.fit_transform(self.M_cause)
            M_effect_pca=pca_effect.fit_transform(self.M_effect)
            M_cond_pca=pca_cond.fit_transform(self.M_cond)
            
            self.dists_cause=self.get_distance_vanilla(M_cause_pca)
            self.dists_effect=self.get_distance_vanilla(M_effect_pca)
            self.dists_cond=self.get_distance_vanilla(M_cond_pca)

            # starting from the effect, first map to reconstruct the condition and directly the cause
            for t_tar in range(self.M_effect.shape[0]):
                # -------The condition manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the condition manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cond_reconst[t_tar]=np.sum(w[:,np.newaxis]*self.M_cond[nearest_time_indices], axis=0)

                # -------The cause manifold reconstruction from the effect -------
                # get the nearest distances of the target point t_tar on the effect manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_effect, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the effect manifold
                self.M_cause_reconst1[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)

            self.dists_cond_reconst=self.get_distance_vanilla(self.M_cond_reconst)

            # starting from the reconstructed condition, map to reconstruct the cause
            for t_tar in range(self.M_cond.shape[0]):
                # -------The cause manifold reconstruction from the reconstructed condition -------
                # get the nearest distances of the target point t_tar on the reconstructed condition manifold
                nearest_time_indices, nearest_distances=self.get_nearest_distances(self.dists_cond_reconst, t_tar, self.knn)
                # get the weights of the nearest neighbors on the cause manifold
                v=np.exp(-nearest_distances/np.max([1e-10, nearest_distances[0]]))
                w=v/np.sum(v)
                # get the reconstruction of the target point t_tar with corresponding points on the reconstructed condition manifold
                self.M_cause_reconst2[t_tar]=np.sum(w[:,np.newaxis]*self.M_cause[nearest_time_indices], axis=0)

            return self.M_cause_reconst1, self.M_cause_reconst2

    
    def causality(self):
        """ Causality scores:
        Correlation based:
            1. direct correlation between M_cause and M_cause_reconst1;
            2. partial correlation between M_cause and M_cause_reconst1 given M_cause_reconst2.
            3. the ratio of ParCorr over DirectCorr.
            
        Error based:
            1. direct error between M_cause and M_cause_reconst1;
            2. indirect error between M_cause and M_cause_reconst2.
            3. the ratio of IndirectError over DirectError.
        """

        # get the reconstructions of the two manifolds
        M_cause_reconst1, M_cause_reconst2=self.predict_manifolds()

        # get the causality score (error)
        sc1_error=np.mean(np.sqrt(np.sum((self.M_cause-M_cause_reconst1)**2, axis=1)))
        sc2_error=np.mean(np.sqrt(np.sum((self.M_cause-M_cause_reconst2)**2, axis=1)))
        ratio_error=sc2_error/sc1_error

        # get the causality score (pearson correlation) - average over each emd dimension
        # direct correlation
        sc1_corr=corr(self.M_cause, M_cause_reconst1)
        sc1_corr=np.mean(np.abs(sc1_corr))
        # partial correlation conditioned on M_cause_reconst2
        sc2_corr=partial_corr(self.M_cause, M_cause_reconst1, M_cause_reconst2)
        sc2_corr=np.nanmean(np.abs(sc2_corr))
        ratio_corr=sc2_corr/sc1_corr

        # the causality score of r2
        sc1_r2=1-np.sum((self.M_cause-M_cause_reconst1)**2)/np.sum((self.M_cause-np.mean(self.M_cause))**2)
        sc2_r2=1-np.sum((self.M_cause-M_cause_reconst2)**2)/np.sum((self.M_cause-np.mean(self.M_cause))**2)
        ratio_r2=sc2_r2/sc1_r2

        return sc1_error, sc2_error, ratio_error, sc1_corr, sc2_corr, ratio_corr, sc1_r2, sc2_r2, ratio_r2
    


# model structure inspired by RESIT: https://lingam.readthedocs.io/en/latest/_modules/lingam/resit.html#RESIT
# Phase 1: Causal order determination with bivariate convergent cross mapping (CM)
# Phase 2: Elimination of redundant edges with multivariate partial convergent cross mapping (multiPCM)

class MXMap:
    """ The class for the Multivar cross (X) MAPping model.
    Two-phase framework:
    1. Initial causal graph determination with cross mapping (CM) - do bivariate CM exhaustively on all the variables.
    2. Elimination of redundant edges with partial CM.
    
    Parameters:
    df: pandas DataFrame
        the data with each column as a variable.
    score_type: str, default 'err', determine which score to use for causal order determination and edge elimination.
        'err' for error, 'corr' for correlation. 
    tau: int, default 2
        time delay for time delay embedding.
    emd: int, default 8
        embedding dimension for time delay embedding.
    pcm_thres: float, default 0.4
        threshold for PCM, 
        If using correlation score: if threshold is smaller than the value, do not remove the edge;
                            else remove the edge.
        If using error score: if threshold is greater than the value, do not remove the edge;
                            else remove the edge.
    **kwargs: dict for additional CM and PCM parameters, whether to use PCA on embedding first 
    before doing kNN during cross mapping."""

    def __init__(self, df, score_type='corr', tau=2, emd=8, pcm_thres=0.5, **kwargs):
        
        self.df = df # the dataframe, extract the column names or indices, determine the causal graph
        self.kwargs = kwargs # dictionary of other parameters, including the CM and PCM parameters
        
        if score_type not in ['corr', 'err', 'r2']:
            raise ValueError('score_type must be either "corr" or "err" or "r2"')
        self.score_type = score_type

        self.tau = tau # time delay for time delay embedding
        self.emd = emd

        self.pcm_thres = pcm_thres # threshold for PCM edge removal

        self.n = df.shape[1] # number of variables
        self.var_names = df.columns # variable names
        self.var_indices = np.arange(self.n) # pool of variable indices

        self.adj_matrix = None # adjacency matrix of the causal graph

        # to save the score stats of the two phases for information
        self.phase1_stats = {}
        self.phase2_stats = {}

    
    def fit(self):
        """ Fit the multivarCM model.
        Returns:
        ch: dict
            the children of each variable.
"""
        ch=self._initial_causal_graph()
        ch=self._eliminate_edges(ch)
        self.ch=ch


        return ch
    
    def get_adj_matrix(self):
        """ Get the adjacency matrix of the causal graph. (to be called after fitting)
        Returns:
        adj_matrix: numpy array
            the adjacency matrix of the causal graph."""
        return self.adj_matrix

    def draw_graph(self, save_path=None):
        """ Draw the causal graph.
        Args:
        ch: dict
            the children of each variable."""
        ch=self.ch
        dot = graphviz.Digraph()
        for k in ch:
            for c in ch[k]:
                cause_name = self.var_names[k]
                effect_name = self.var_names[c]
                dot.edge(cause_name, effect_name)

        # view and save the graph
        if save_path is not None:
            dot.render(os.path.join(save_path, 'causal_graph'), format='png', view=True)

        return dot

    def _initial_causal_graph(self):
        # exhaustive bivariate search for initial causal graph (doesn't distinguish between direct and indirect)
        S=self.var_indices.copy() # pool of variable indices, start with all variables

        # initialize adjacency matrix
        self.adj_matrix=np.zeros((self.n,self.n))

        ch={} # dictionary to store the children of each variable
        
        sc_ratio_stats={}
        for i in range(self.n): # cause

            # initialize children of current var
            ch[i]=[]
            #  store the score stats
            sc_ratio_stats[i]={}

            # do not test redundant pairs
            for j in S[S>i]: # effect
                # cross map between the current variable i and the candidate child j
                # determine whether the edge between i and j is redundant
                cause_ind=[i]
                effect_ind=[j]
                cause_list=self.var_names[cause_ind].tolist()
                effect_list=self.var_names[effect_ind].tolist()

                # key in dictionary to store the stats ("cause -> effect | conds")
                phase1_stats_key=f'causes_{cause_ind} -> effect_{effect_ind}'

                # create the CM_simplex object
                cm=CM_simplex(self.df,cause_list,effect_list,self.tau,self.emd,**self.kwargs)
                output=cm.causality() # order: sc1_err, sc2_err, sc1_corr, sc2_corr, sc1_r2, sc2_r2


                del cm

                # store the stats
                self.phase1_stats[phase1_stats_key] = output

                if self.score_type=='err': 
                    ratio=output[0]/output[1]
                if self.score_type=='corr':
                    ratio=output[2]/output[3]
                if self.score_type=='r2':
                    ratio=output[4]/output[5]

                # store the ratio
                sc_ratio_stats[i][j]=ratio

        # link all the variable pairs if the ratio is greater than 1 (corr, r1) or smaller than 1 (err)
        for i in range(self.n):
            if self.score_type=='err':
                for j in S[S>i]:
                    if sc_ratio_stats[i][j]<1:
                        ch[i].append(j)
                        self.adj_matrix[i,j]=1
                    else:
                        ch[j].append(i)
                        self.adj_matrix[j,i]=1
            else: # if corr or r2
                for j in S[S>i]:
                    # access output from phase1 stats
                    output=self.phase1_stats[f'causes_{[i]} -> effect_{[j]}']
                    if sc_ratio_stats[i][j]>1:
                        # condition for not establishing the edge, if the score (corr or r2) is too small
                        if self.score_type=='corr':
                            if output[2]<0.5 and output[3]<0.5:
                                continue
                        if self.score_type=='r2':
                            if output[4]<0.5 and output[5]<0.5:
                                continue
                        ch[i].append(j)
                        self.adj_matrix[i,j]=1
                    else:
                        ch[j].append(i)
                        if self.score_type=='corr':
                            if output[2]<0.5 and output[3]<0.5:
                                continue
                        if self.score_type=='r2':
                            if output[4]<0.5 and output[5]<0.5:
                                continue
                        self.adj_matrix[j,i]=1

        return ch     
                
        
    def _eliminate_edges(self, ch):
        """ The second step: eliminate redundant edges.
        Use the score_type to determine which returned scores from PCM model to use.
        * Note my definition of ordering is from sink to top.
        
        For each pairwise edge that has other variables in between, this might be indirecto causality"""

        # Multi PCM
        list_to_remove=[] # will be used to store tuples of edges (cause, effect) to remove

        for i in range(self.n):
            for j in ch[i]: # children of i
                # check if a causal path (from adjacency matrix) can be established between i and j
                # if there is a path, do PCM to determine if it is a indirect causation
                    # if it is, remove the edge between i and j
                    # if it is not, keep the edge between i and j
                # if there is no path, keep the edge between i and j
                
                # bool, list_var_on_path = has_path(self.adj_matrix, i, j)
                bool, list_var_on_path = find_longest_path(self.adj_matrix, i, j)

                if bool:
                    # create the PCM object
                    # cause_list=self.var_names[[i]].tolist()
                    # effect_list=self.var_names[[j]].tolist()
                    # conds_list=[self.var_names[k] for k in list_var_on_path]

                    cause_ind=[i]
                    effect_ind=[j]
                    # conditions: list_var_on_path - i - j
                    conds_ind=[k for k in list_var_on_path if k!=i and k!=j]

                    # skip if there are no conditions
                    if len(conds_ind)==0:
                        continue

                    cause_list=self.var_names[cause_ind].tolist()
                    effect_list=self.var_names[effect_ind].tolist()
                    conds_list=self.var_names[conds_ind].tolist() 

                    # key in dictionary to store the stats ("cause -> effect | conds")
                    phase2_stats_key=f'causes_{cause_ind} -> effect_{effect_ind} | conds_{conds_ind}'

                    pcm=PCM_simplex(self.df,cause_list,effect_list,conds_list,self.tau,self.emd,**self.kwargs)
                    output=pcm.causality() # sc1_error, sc2_error, ratio_error, sc1_corr, sc2_corr, ratio_corr, sc1_r2, sc2_r2, ratio_r2
                    del pcm

                    # store the stats
                    self.phase2_stats[phase2_stats_key] = output


                    if self.score_type=='err':
                        if output[2]>self.pcm_thres:
                            list_to_remove.append((i,j))
                    if self.score_type=='corr':
                        if output[5]<self.pcm_thres:
                            list_to_remove.append((i,j))
                    if self.score_type=='r2':
                        if output[8]<self.pcm_thres:
                            list_to_remove.append((i,j))
                else:
                    continue

        # remove the edges
        for edge in list_to_remove:
            i,j=edge
            ch[i].remove(j)
            self.adj_matrix[i,j]=0

        return ch

def find_longest_path(adj_matrix, i, j):
    """Find the longest path of length >= 3 between two variables i and j.

    Args:
        adj_matrix: numpy array
            the adjacency matrix of the causal graph.
        i: int
            the index of the cause variable.
        j: int
            the index of the effect variable.
    
    Returns:
        bool: True if there is a path of length >= 3 between i and j, False otherwise.
        list: The longest path of nodes if a valid path exists, otherwise None.
    """
    n = adj_matrix.shape[0]
    stack = [(i, [i])]  # Stack stores tuples of (current node, path)
    longest_path = []  # Track the longest valid path
    
    while stack:
        node, path = stack.pop()

        if node == j and len(path) >= 3:  # Check if the current path is valid
            if len(path) > len(longest_path):  # Update if it's the longest valid path
                longest_path = path
        
        for k in range(n):
            if adj_matrix[node, k] == 1 and k not in path:  # Avoid revisiting nodes in the same path
                stack.append((k, path + [k]))  # Append the new path

    if longest_path:
        return True, longest_path  # Return the longest valid path
    else:
        return False, None  # No valid path found

        