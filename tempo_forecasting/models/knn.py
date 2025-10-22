from .base_model import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Any

from tempo_forecasting.utils.logging_utils import logger

class KNNModel(BaseModel):
    """
    A K Nearest Neighbors model for time series forecasting, based on the 2015 paper
    'A Study of the Use of Complexity Measures in the Similarity Search Process 
    Adopted by kNN Algorithm for Time Series Prediction'
    by Antonio Rafael Sabino Parmezan, Gustavo E. A. P. A. Batista'

    Attributes:
        target_y (str): The name of the target variable to be forecasted.
        date_col (str): The name of the column containing date information.
        fitted_vals (np.ndarray): Array of the training values. KNN has no true fitted vals.
        default_model_params (dict): A dictionary which stores the default parameters for the model.
        len_q (int): Length of the query window, i.e. length of time period the model will be finding matches for 
        k (int): Number of neighbors
        forecast_chunk_size (int): Length of each forecast step derived from one set of neighbors. The algorithm will
            generate a forecast of forecast_chunk_size length, then use those forecasted values in the next query
        forecast_horizon (int): Final length of the output forecast when using .predict()
        subsequences: Training data, broken into overlapping rolling windows of length len_q
        subsequence_nexts: The data (of length forecast_chunk_size) immediately after each subsequence
        standardized_subsequences: Z-score standardized subsequences
        standardized_nexts: subsequence_nexts, standardized with the corresponding subsequence's mean and std
        nn_starting_indices: The starting indices of each neighbor found during forecasting


    Methods:
        _build_subsequences:
            Break training data (1D np.ndarray) into rolling window subsequences of length len_q
        _calculate_complexity_estimates:
            Calculate complexity estimate for each row of a 2D np.ndarray
        _calculate_complexity_correction_factors:
            Calculate complexity correction factors row by row between two 2D np.ndarrays
        _calculate_euclidean_distances:
            Calculate euclidean distances row by row between two 2D np.ndarrays
        _calculate_complexity_invariant_distances:
            Calculate complexity invariant distances between a query array (size 1xn) and 
            an array of subsequences (size mxn)
        fit: 
            Method to fit the model to the given training data.
        predict:
            Method to generate predictions for the specified future dates.
    """
    def __init__(self, 
                 target_y: str,
                 date_col: str
                 ):
        """
        Initializes the KNNModel with the specified target and date columns.

        Parameters:
            target_y (str): The name of the target variable to be forecasted.
            date_col (str): The name of the column containing date information.
        """
        super().__init__("knn",
                         date_col=date_col, 
                         target_y=target_y
                         )
        
        self.len_q = None
        self.forecast_chunk_size = None
        self.k = None
        self.forecast_horizon = None

        self.fitted_vals = None

        self.subsequences = None
        self.subsequence_nexts = None
        self.standardized_s = None
        self.standardized_nexts = None

        self.nn_starting_indices = None
        logger.debug(f"Initialized KNNModel with target_y: '{target_y}' and date_col: '{date_col}'")

    # Data Prep
    def _build_subsequences(self, 
                            data: np.ndarray, 
                            n_steps: int) -> np.ndarray:
        """
        Constructs overlapping, rolling-window subsequences of a specified length from the provided data.

        This function takes a dataset and generates overlapping subsequences of a defined 
        length, `n_steps`. The resulting subsequences can be useful for time series 
        analysis, feature extraction, or model training.

        Parameters:
            data (np.ndarray): The input array of data from which to extract subsequences.
                            It is assumed to be a one-dimensional array.

            n_steps (int): The length of each subsequence to be created. It should be 
                        a positive integer indicating how many consecutive data points 
                        to include in each subsequence.

        Returns:
            np.ndarray: A 2D array of shape (num_subsequences, n_steps) containing 
                        the extracted subsequences, where `num_subsequences` is determined 
                        by the length of the input data and `n_steps`.
        """

        subsequences = np.array([])

        len_index = len(data)-n_steps+1
        for i in range(len_index):
            subsequences = np.append(subsequences,data[i:i+n_steps])
        subsequences = subsequences.reshape((len_index,n_steps))
        return subsequences

    # Complexity Invariant Distance
    def _calculate_complexity_estimates(self, 
                                        arr: np.ndarray) -> np.ndarray:
        """
        Calculates complexity estimates from a given array.

        This function computes the complexity estimates by calculating the square root 
        of the sum of squared differences between consecutive elements along the specified 
        axis of the input array. This can be useful for assessing the variability or 
        complexity of the data represented in the array.

        Parameters:
            arr (np.ndarray): An input array containing the data from which complexity 
                            estimates are to be calculated. 

        Returns:
            np.ndarray: An array of complexity estimates, where each element represents 
                        the calculated complexity for the corresponding row of the input array.
        """

        ce = np.sqrt(np.sum(np.diff(arr)**2,axis=1, keepdims=True))
        return ce

    def _calculate_complexity_correction_factors(self, 
                                                 ce_a: np.ndarray,
                                                 ce_b: np.ndarray) -> np.ndarray:
        """
        Calculates complexity correction factors based on two mx1 arrays of complexity estimates.

        This method computes the complexity correction factors (CCF) by taking the maximum 
        and minimum of two complexity estimates, `ce_a` and `ce_b`. It ensures numerical stability 
        by adding a small epsilon value to avoid division by zero. 

        Parameters:
            ce_a (np.ndarray): The first array of complexity estimates, mx1.
            ce_b (np.ndarray): The second array of complexity estimates, mx1.

        Returns:
            np.ndarray: An mx1 array of complexity correction factors, where each factor is
                        the ratio of the maximum complexity estimate to the minimum complexity 
                        estimate, adjusted for numerical stability. The ith row contains the 
                        complexity correction factor derived by comparing the ith elements of 
                        ce_a and ce_b.
        """

        max_ce = np.max((ce_a,ce_b), axis=0) + np.finfo("float32").eps
        min_ce = np.min((ce_a,ce_b), axis=0) + np.finfo("float32").eps
        ccf = max_ce/min_ce    

        return ccf

    def _calculate_euclidean_distances(self, 
                                       a: np.ndarray,
                                       b: np.ndarray) -> np.ndarray:
        """
        Calculates the Euclidean distances between two sets of points.

        This method computes the Euclidean distance between each pair of corresponding 
        points in two arrays (a and b). The distance is calculated as the square root 
        of the sum of the squared differences along each dimension.

        Parameters:
            a (np.ndarray): A 2D array of shape (n_samples, n_features) representing 
                            the first set of points.
            b (np.ndarray): A 2D array of shape (n_samples, n_features) representing 
                            the second set of points. This array must have the same 
                            shape as `a`.

        Returns:
            np.ndarray: A 2D array of shape (n_samples, 1) containing the Euclidean 
                        distances for each pair of points. Each entry in the array 
                        corresponds to the distance between the respective points in 
                        arrays `a` and `b`.
        """

        euclidean_dist = np.sqrt(np.sum((a - b) ** 2,axis=1,keepdims=True))
        return euclidean_dist

    def _calculate_complexity_invariant_distances(self, 
                                                  standardized_q: np.ndarray,
                                                  standardized_s: np.ndarray) -> np.ndarray:
        """
        Calculates complexity-invariant distances based on the provided standardized 
        quantities and similarity measures.

        This method first computes Euclidean distances between the standardized subsequences 
        and the query sequence. Then, it multiplies the Euclidean distance by a complexity correction factor,
        which is itself derived from a comparison complexity estimates for the subsequences and query sequence.

        Parameters:
            standardized_q (np.ndarray): A 1D array containing standardized values for the 
                                        quantity of interest.
            standardized_s (np.ndarray): A 2D array containing standardized similarity measures 
                                        across observations.

        Returns:
            np.ndarray: A 1D array of computed complexity-invariant distances, reflecting adjustments
                        based on complexity estimates and correction factors associated with the provided 
                        standardized inputs.
        """

        ed = self._calculate_euclidean_distances(standardized_s,
                                        np.ones(standardized_s.shape)*standardized_q)

        ce_s = self._calculate_complexity_estimates(standardized_s)
        ce_q = self._calculate_complexity_estimates(standardized_q)

        ccf = self._calculate_complexity_correction_factors(ce_s,
                                                    np.ones(ce_s.shape)*ce_q)

        cid = ed*ccf
        return cid

    def fit(self, 
            train_data: pd.DataFrame, 
            model_param_dict: Dict[str,Any]
            ) -> None:
        """
        Prepares the knn model by initializing the historical data.

        Here, "fit" means "build subsequences and standardize"
        There is no true "fit" for a knn model
        Need fitted vals to play nice with tempo_forecasting though

        Parameters:
            train_data (pd.DataFrame): A DataFrame containing the training data.
                Must include the target variable (`target_y`).
            model_param_dict (dict, optional): A dictionary of parameters to specify for the model.
                Defaults to self.default_params
        """
        logger.debug("Starting model fitting.")

        final_params = self._combine_and_categorize_params(model_param_dict)
        self.k = final_params["custom_params"]["k"]

        try:
            self.fitted_vals = np.array(train_data[self.target_y])
            self.len_q = final_params["custom_params"]["len_q"]
            self.forecast_chunk_size = final_params["custom_params"]["forecast_chunk_size"]

            # Break training data into rolling window subsequences of length len_q
            # and save the periods of time AFTER to form predictions
            subsequences_with_next = self._build_subsequences(self.fitted_vals,self.len_q+self.forecast_chunk_size)
            self.subsequences = subsequences_with_next[:,:-self.forecast_chunk_size]
            self.subsequence_nexts = subsequences_with_next[:,-self.forecast_chunk_size:]

            # Standardize the subsequences
            # saving their mean and standard deviations for later
            s_mu = self.subsequences.mean(axis=1, keepdims=True)
            s_std = self.subsequences.std(axis=1, keepdims=True)
            # to avoid dividing by 0
            s_std[s_std == 0] = 1 

            self.standardized_s = (self.subsequences - s_mu)/s_std
            self.standardized_nexts = (self.subsequence_nexts - s_mu)/s_std
            logger.debug("Model fitting completed successfully.")

        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise


    def predict(self, 
                test_data: pd.DataFrame) -> np.ndarray:
        """
        Generates forecasts for the time period given in the 
        test data using the moving average.

        Parameters:
            test_data (pd.DataFrame): A DataFrame containing the test data.
                The index must be a `DatetimeIndex`.

        Returns:
            np.ndarray: A 1-dimensional array of forecasted values corresponding to 
                the indices of the test data.
        """
        logger.debug("Starting prediction.")
        try:
            self.forecast_horizon = len(test_data.index)

            # Iteratively build forecast
            forecast = np.array([])
            nn_starting_indices = np.array([])
            n_forecast_chunks = int(np.ceil(self.forecast_horizon/self.forecast_chunk_size))
            for i in range(n_forecast_chunks):
                # build and standardize query
                # wrap query in np.array([]) to have appropriate dimensions
                query = np.array([np.append(self.fitted_vals,forecast)[-self.len_q:]]) 
                q_mu = query.mean(axis=1, keepdims=True)
                q_std = query.std(axis=1, keepdims=True)
                q_std[q_std == 0] = 1 
                standardized_q = (query-q_mu)/q_std

                # Calculate Distances
                cid = self._calculate_complexity_invariant_distances(standardized_q,self.standardized_s)

                # Find Nearest Neighbors
                # subsequences that overlap the query or an existing neighbor are ineligible
                trivial_match_mask = np.ones(cid.shape)
                query_train_overlap = max(self.len_q + self.forecast_chunk_size - forecast.shape[0],0)
                if query_train_overlap > 0:
                    trivial_match_mask[-query_train_overlap:] = float("inf")
                trivial_match_masked_cid = trivial_match_mask * cid

                nn_indices = np.array([])
                for i in range(self.k):
                    nn_index = np.argmin(trivial_match_masked_cid)
                    nn_indices = np.append(nn_indices,[nn_index])
                    trivial_match_mask[nn_index-(self.len_q-1):nn_index+self.len_q] = float("inf")
                    trivial_match_masked_cid = trivial_match_masked_cid*trivial_match_mask

                nn_indices = nn_indices.astype("int")
                nn_starting_indices = np.append(nn_starting_indices,[nn_indices])

                # Compile NN into prediction
                pred = np.mean(self.standardized_nexts[nn_indices],axis=0)*q_std + q_mu
                forecast = np.append(forecast,[pred])

            self.nn_starting_indices = nn_starting_indices # for QA purposes
            forecast = np.clip(forecast[:self.forecast_horizon], a_min=0, a_max=None)

            logger.debug("Prediction completed successfully.")
            if np.any(forecast < 0):
                logger.warning("Negative forecast values were clipped to 0.")

            return forecast

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise