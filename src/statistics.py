import numpy as np 

import pandas as pd 

from scipy import stats 

from scipy.stats import norm, t, chi2, f 

from typing import List, Tuple, Dict, Union, Optional 

import warnings 

 

class StatisticalAnalyzer: 

    """ 

    Comprehensive statistical analysis toolkit for first-year statistics. 

    """ 

     

    def __init__(self, data: pd.DataFrame): 

        """ 

        Initialize with a pandas DataFrame. 

         

        Args: 

            data: Cleaned pandas DataFrame 

        """ 

        self.data = data 

        self.numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns 

        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns 

     

     

# ==================== DESCRIPTIVE STATISTICS ==================== 

     

    def descriptive_stats(self, column: str) -> Dict: 

        """ 

        Calculate comprehensive descriptive statistics for a numeric column. 

         

        Args: 

            column: Column name 

             

        Returns: 

            Dictionary with all descriptive statistics 

        """ 

        data = self.data[column].dropna() 

         

        stats_dict = { 

            'column': column, 

            'n': len(data), 

            'mean': data.mean(), 

            'median': data.median(), 

            'mode': data.mode()[0] if not data.mode().empty else np.nan, 

            'std': data.std(), 

            'variance': data.var(), 

            'min': data.min(), 

            'max': data.max(), 

            'range': data.max() - data.min(), 

            'q1': data.quantile(0.25), 

            'q3': data.quantile(0.75), 

            'iqr': data.quantile(0.75) - data.quantile(0.25), 

            'skewness': data.skew(), 

            'kurtosis': data.kurtosis(), 

            'cv': data.std() / data.mean() if data.mean() != 0 else np.nan, 

            'missing': self.data[column].isnull().sum() 

        } 

         

        return stats_dict 

     

    def all_descriptive_stats(self) -> pd.DataFrame: 

        """ 

        Calculate descriptive statistics for ALL numeric columns. 

         

        Returns: 

            DataFrame with descriptive statistics for each column 

        """ 

        results = [] 

        for col in self.numeric_cols: 

            results.append(self.descriptive_stats(col)) 

         

        return pd.DataFrame(results) 

     

# ==================== NORMALITY TESTS ==================== 

     

    def shapiro_wilk_test(self, column: str, alpha: float = 0.05) -> Dict: 

        """ 

        Perform Shapiro-Wilk test for normality. 

         

        Args: 

            column: Column name 

            alpha: Significance level (default 0.05) 

             

        Returns: 

            Dictionary with test results 

        """ 

        data = self.data[column].dropna() 

         

        if len(data) < 3: 

            return { 

                'column': column, 

                'statistic': np.nan, 

                'p_value': np.nan, 

                'normal': False, 

                'error': 'Sample size too small (n < 3)' 

            } 

         

        statistic, p_value = stats.shapiro(data) 

        normal = p_value > alpha 

         

        return { 

            'column': column, 

            'statistic': statistic, 

            'p_value': p_value, 

            'normal': normal, 

            'interpretation': "Normal" if normal else "Not normal", 

            'note': f"p = {p_value:.4f} {'>' if normal else '<='} α = {alpha}" 

        } 

     

    def kolmogorov_smirnov_test(self, column: str, dist: str = 'norm', alpha: float = 0.05) -> Dict: 

        """ 

        Perform Kolmogorov-Smirnov test for goodness of fit. 

         

        Args: 

            column: Column name 

            dist: Distribution to test ('norm', 'expon', 'uniform') 

            alpha: Significance level 

             

        Returns: 

            Dictionary with test results 

        """ 

        data = self.data[column].dropna() 

         

        if dist == 'norm': 

            # Standardize data for normal distribution 

            data_std = (data - data.mean()) / data.std() 

            statistic, p_value = stats.kstest(data_std, 'norm') 

        elif dist == 'expon': 

            # Fit exponential distribution 

            loc, scale = stats.expon.fit(data) 

            statistic, p_value = stats.kstest(data, 'expon', args=(loc, scale)) 

        elif dist == 'uniform': 

            statistic, p_value = stats.kstest(data, 'uniform', args=(data.min(), data.max() - data.min())) 

        else: 

            raise ValueError(f"Unsupported distribution: {dist}") 

         

        fit_good = p_value > alpha 

         

        return { 

            'column': column, 

            'distribution': dist, 

            'statistic': statistic, 

            'p_value': p_value, 

            'good_fit': fit_good, 

            'interpretation': "Good fit" if fit_good else "Poor fit" 

        } 

     

    def all_normality_tests(self, alpha: float = 0.05) -> pd.DataFrame: 

        """ 

        Run Shapiro-Wilk test on all numeric columns. 

         

        Returns: 

            DataFrame with normality test results 

        """ 

        results = [] 

        for col in self.numeric_cols: 

            results.append(self.shapiro_wilk_test(col, alpha)) 

         

        return pd.DataFrame(results) 

     

# ==================== HYPOTHESIS TESTING ==================== 

     

    def t_test(self, column1: str, column2: Union[str, float],  

               paired: bool = False, alternative: str = 'two-sided') -> Dict: 

        """ 

        Perform t-test (one-sample, independent, or paired). 

         

        Args: 

            column1: First column name 

            column2: Second column name OR value for one-sample test 

            paired: Whether it's a paired test 

            alternative: 'two-sided', 'less', or 'greater' 

             

        Returns: 

            Dictionary with test results 

        """ 

        data1 = self.data[column1].dropna() 

         

        if isinstance(column2, (int, float)): 

            # One-sample t-test 

            statistic, p_value = stats.ttest_1samp(data1, column2, alternative=alternative) 

             

            return { 

                'test': 'One-sample t-test', 

                'variable': column1, 

                'mean': data1.mean(), 

                'null_value': column2, 

                'statistic': statistic, 

                'p_value': p_value, 

                'df': len(data1) - 1, 

                'significant': p_value < 0.05, 

                'interpretation': f"Mean significantly differs from {column2}" if p_value < 0.05 else f"Mean not significantly different from {column2}" 

            } 

        else: 

            # Two-sample t-test 

            data2 = self.data[column2].dropna() 

             

            # Check if paired 

            if paired: 

                if len(data1) != len(data2): 

                    return {'error': 'Paired test requires equal sample sizes'} 

                statistic, p_value = stats.ttest_rel(data1, data2, alternative=alternative) 

                test_type = 'Paired t-test' 

            else: 

                # Check equal variances for independent t-test 

                if stats.levene(data1, data2).pvalue > 0.05: 

                    # Equal variance assumed 

                    statistic, p_value = stats.ttest_ind(data1, data2, equal_var=True, alternative=alternative) 

                else: 

                    # Equal variance NOT assumed (Welch's t-test) 

                    statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False, alternative=alternative) 

                test_type = 'Independent t-test' 

             

            return { 

                'test': test_type, 

                'variable1': column1, 

                'variable2': column2, 

                'mean1': data1.mean(), 

                'mean2': data2.mean(), 

                'mean_diff': data1.mean() - data2.mean(), 

                'statistic': statistic, 

                'p_value': p_value, 

                'df': len(data1) + len(data2) - 2 if not paired else len(data1) - 1, 

                'significant': p_value < 0.05, 

                'interpretation': f"Significant difference between {column1} and {column2}" if p_value < 0.05 else f"No significant difference between {column1} and {column2}" 

            } 

     

    def anova_test(self, numeric_col: str, categorical_col: str) -> Dict: 

        """ 

        Perform one-way ANOVA test. 

         

        Args: 

            numeric_col: Numeric column name 

            categorical_col: Categorical column name (groups) 

             

        Returns: 

            Dictionary with ANOVA results 

        """ 

        groups = [] 

        group_names = self.data[categorical_col].unique() 

         

        for group in group_names: 

            group_data = self.data[self.data[categorical_col] == group][numeric_col].dropna() 

            if len(group_data) > 0: 

                groups.append(group_data) 

         

        if len(groups) < 2: 

            return {'error': 'Need at least 2 groups for ANOVA'} 

         

        f_statistic, p_value = stats.f_oneway(*groups) 

         

        return { 

            'test': 'One-way ANOVA', 

            'numeric_variable': numeric_col, 

            'categorical_variable': categorical_col, 

            'groups': len(groups), 

            'f_statistic': f_statistic, 

            'p_value': p_value, 

            'significant': p_value < 0.05, 

            'interpretation': f"Significant difference among groups" if p_value < 0.05 else f"No significant difference among groups" 

        } 

     

    def chi_square_test(self, col1: str, col2: str) -> Dict: 

        """ 

        Perform chi-square test of independence. 

         

        Args: 

            col1: First categorical column 

            col2: Second categorical column 

             

        Returns: 

            Dictionary with chi-square results 

        """ 

        contingency = pd.crosstab(self.data[col1], self.data[col2]) 

        statistic, p_value, dof, expected = stats.chi2_contingency(contingency) 

         

        return { 

            'test': 'Chi-square Test of Independence', 

            'variable1': col1, 

            'variable2': col2, 

            'chi2_statistic': statistic, 

            'p_value': p_value, 

            'degrees_of_freedom': dof, 

            'significant': p_value < 0.05, 

            'interpretation': f"Variables are dependent" if p_value < 0.05 else f"Variables are independent" 

        } 

     

# ==================== CONFIDENCE INTERVALS ==================== 

     

    def confidence_interval(self, column: str, confidence: float = 0.95) -> Dict: 

        """ 

        Calculate confidence interval for a numeric column. 

         

        Args: 

            column: Column name 

            confidence: Confidence level (default 0.95) 

             

        Returns: 

            Dictionary with confidence interval results 

        """ 

        data = self.data[column].dropna() 

        n = len(data) 

        mean = data.mean() 

        std = data.std() 

        se = std / np.sqrt(n) 

         

        # For n >= 30, use z-distribution; otherwise use t-distribution 

        if n >= 30: 

            z_score = stats.norm.ppf((1 + confidence) / 2) 

            margin = z_score * se 

            dist_used = 'z (normal)' 

        else: 

            t_score = stats.t.ppf((1 + confidence) / 2, df=n-1) 

            margin = t_score * se 

            dist_used = 't (Student\'s)' 

         

        lower = mean - margin 

        upper = mean + margin 

         

        return { 

            'column': column, 

            'mean': mean, 

            'std': std, 

            'n': n, 

            'confidence_level': confidence, 

            'se': se, 

            'distribution': dist_used, 

            'lower_bound': lower, 

            'upper_bound': upper, 

            'margin_of_error': margin, 

            'interpretation': f"We are {confidence*100}% confident that the true mean lies between {lower:.3f} and {upper:.3f}" 

        } 

     

    def all_confidence_intervals(self, confidence: float = 0.95) -> pd.DataFrame: 

        """ 

        Calculate confidence intervals for all numeric columns. 

         

        Returns: 

            DataFrame with confidence intervals 

        """ 

        results = [] 

        for col in self.numeric_cols: 

            results.append(self.confidence_interval(col, confidence)) 

         

        return pd.DataFrame(results) 

     

# ==================== BOOTSTRAP ==================== 

     

    def bootstrap_ci(self, column: str, statistic: callable = np.mean,  

                     n_bootstrap: int = 10000, confidence: float = 0.95) -> Dict: 

        """ 

        Calculate bootstrap confidence interval for any statistic. 

         

        Args: 

            column: Column name 

            statistic: Function to bootstrap (e.g., np.mean, np.median) 

            n_bootstrap: Number of bootstrap samples 

            confidence: Confidence level 

             

        Returns: 

            Dictionary with bootstrap results 

        """ 

        data = self.data[column].dropna().values 

        n = len(data) 

         

        # Generate bootstrap samples 

        bootstrap_stats = [] 

        for _ in range(n_bootstrap): 

            sample = np.random.choice(data, size=n, replace=True) 

            bootstrap_stats.append(statistic(sample)) 

         

        # Calculate confidence interval 

        alpha = 1 - confidence 

        lower_percentile = alpha / 2 * 100 

        upper_percentile = (1 - alpha / 2) * 100 

         

        lower = np.percentile(bootstrap_stats, lower_percentile) 

        upper = np.percentile(bootstrap_stats, upper_percentile) 

         

        return { 

            'column': column, 

            'statistic': statistic.__name__, 

            'original_value': statistic(data), 

            'n_bootstrap': n_bootstrap, 

            'confidence_level': confidence, 

            'lower_bound': lower, 

            'upper_bound': upper, 

            'bootstrap_mean': np.mean(bootstrap_stats), 

            'bootstrap_std': np.std(bootstrap_stats), 

            'interpretation': f"We are {confidence*100}% confident that the true {statistic.__name__} lies between {lower:.3f} and {upper:.3f}" 

        } 

 

# ==================== DISTRIBUTION FITTING ==================== 

     

    def fit_distribution(self, column: str, distributions: List[str] = None) -> Dict: 

        """ 

        Fit various probability distributions to data. 

         

        Args: 

            column: Column name 

            distributions: List of distributions to fit ('norm', 'expon', 'gamma', 'lognorm', 'uniform') 

             

        Returns: 

            Dictionary with distribution fitting results 

        """ 

        data = self.data[column].dropna() 

         

        if distributions is None: 

            distributions = ['norm', 'expon', 'gamma', 'lognorm'] 

         

        results = {} 

         

        for dist_name in distributions: 

            try: 

                if dist_name == 'norm': 

                    params = stats.norm.fit(data) 

                    statistic, p_value = stats.kstest(data, 'norm', args=params) 

                elif dist_name == 'expon': 

                    params = stats.expon.fit(data) 

                    statistic, p_value = stats.kstest(data, 'expon', args=params) 

                elif dist_name == 'gamma': 

                    params = stats.gamma.fit(data) 

                    statistic, p_value = stats.kstest(data, 'gamma', args=params) 

                elif dist_name == 'lognorm': 

                    params = stats.lognorm.fit(data) 

                    statistic, p_value = stats.kstest(data, 'lognorm', args=params) 

                elif dist_name == 'uniform': 

                    params = (data.min(), data.max() - data.min()) 

                    statistic, p_value = stats.kstest(data, 'uniform', args=params) 

                else: 

                    continue 

                 

                results[dist_name] = { 

                    'params': params, 

                    'ks_statistic': statistic, 

                    'p_value': p_value, 

                    'good_fit': p_value > 0.05 

                } 

            except Exception as e: 

                results[dist_name] = {'error': str(e)} 

         

        # Find best fit 

        best_fit = None 

        best_p = -1 

        for dist_name, result in results.items(): 

            if 'p_value' in result and result['p_value'] > best_p: 

                best_p = result['p_value'] 

                best_fit = dist_name 

         

        return { 

            'column': column, 

            'distributions': results, 

            'best_fit': best_fit, 

            'best_p_value': best_p 

        }
