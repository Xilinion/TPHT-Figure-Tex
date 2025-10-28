#!/usr/bin/env python3
"""
Script to calculate and extract in-text data from CSV files for the paper.

This script reads all CSV files from ../csv directory and provides a framework
to calculate specific statistics and values mentioned in the paper text.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import glob

class DataProcessor:
    def __init__(self):
        """Initialize the data processor and load all CSV files."""
        self.csv_dir = Path("../csv")
        self.output_file = "./in_text_data.csv"
        self.latex_file = "./in_text_data.tex"
        
        # Dictionary to store all loaded dataframes
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
        # List to store tuples for output CSV
        self.results: List[Tuple[int, str, Any]] = []
        
        # Load all CSV files
        self.load_csv_files()
        
        # Counter for generating unique IDs
        self.id_counter = 1
    
    def load_csv_files(self):
        """Load all CSV files from the csv directory into dataframes."""
        csv_files = glob.glob(str(self.csv_dir / "*.csv"))
        
        for csv_file in csv_files:
            # Extract filename without extension to use as key
            filename = Path(csv_file).stem
            
            try:
                df = pd.read_csv(csv_file)
                self.dataframes[filename] = df
                print(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"  Columns: {list(df.columns)}")
                print()
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    def add_result(self, name: str, value: Any) -> None:
        """Add a result tuple to the results list."""
        self.results.append((self.id_counter, name, value))
        self.id_counter += 1
    
    def get_dataframe(self, name: str) -> pd.DataFrame:
        """Get a dataframe by name with error checking."""
        if name not in self.dataframes:
            available = ', '.join(self.dataframes.keys())
            raise ValueError(f"Dataframe '{name}' not found. Available: {available}")
        return self.dataframes[name]
    
    # ==========================================================================
    # CALCULATION FUNCTIONS
    # Add your calculation functions below. Each function should call add_result()
    # ==========================================================================
    
    def calculate_throughput_metrics(self):
        """
        Calculate throughput metrics for the throughput.tex section.
        
        Uses: ycsb_results.csv
        Calculates: Various throughput statistics using hash table macros
        """
        df = self.get_dataframe('ycsb_results')
        
        # Filter for entry_id=0 only as specified
        df_filtered = df[df['entry_id'] == 0].copy()
        
        # Object ID to macro mapping
        object_to_macro = {
            6: 'htthree',   # Cuckoo
            7: 'htfour',    # Iceberg
            15: 'htfive',   # Junction
            24: 'htsix',    # Baseline
            17: 'htone',    # TPHT
            20: 'httwo'     # Blast
        }
        
        # === \httwo  (Blast, object_id=20) metrics ===
        httwo_data = df_filtered[df_filtered['object_id'] == 20]
        
        # Average fill throughput for Load phase (case_id=17) - keep raw values
        httwo_load_fill_raw = httwo_data[httwo_data['case_id'] == 17]['fill_throughput (ops/s)'].values[0]
        self.add_result("httwo_avg_fill_throughput", round(httwo_load_fill_raw / 1_000_000, 1))
        
        # Average run throughput across all run phases (case_ids 17-22) - keep raw values
        httwo_run_throughputs_raw = httwo_data[httwo_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
        httwo_avg_run_raw = httwo_run_throughputs_raw.mean()
        self.add_result("httwo_avg_run_throughput", round(httwo_avg_run_raw / 1_000_000, 1))
        
        # === \htone  (TPHT, object_id=17) metrics ===
        htone_data = df_filtered[df_filtered['object_id'] == 17]
        
        # Average fill throughput for Load phase - keep raw values
        htone_load_fill_raw = htone_data[htone_data['case_id'] == 17]['fill_throughput (ops/s)'].values[0]
        self.add_result("htone_avg_fill_throughput", round(htone_load_fill_raw / 1_000_000, 1))
        
        # Average run throughput across all run phases - keep raw values
        htone_run_throughputs_raw = htone_data[htone_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
        htone_avg_run_raw = htone_run_throughputs_raw.mean()
        self.add_result("htone_avg_run_throughput", round(htone_avg_run_raw / 1_000_000, 1))
        
        # === Speedup calculations ===
        # Find fastest baseline for comparison (excluding httwo and htone)
        baseline_objects = [6, 7, 15, 24]  # Cuckoo, Iceberg, Junction, htsix
        
        # Fill throughput comparison - use raw values
        baseline_fill_max_raw = df_filtered[(df_filtered['object_id'].isin(baseline_objects)) & 
                                           (df_filtered['case_id'] == 17)]['fill_throughput (ops/s)'].max()
        
        httwo_fill_speedup = httwo_load_fill_raw / baseline_fill_max_raw
        htone_fill_speedup = htone_load_fill_raw / baseline_fill_max_raw
        
        self.add_result("httwo_fill_speedup", round(httwo_fill_speedup, 2))
        self.add_result("htone_fill_speedup_percent", round(htone_fill_speedup * 100, 1))
        
        # Run throughput comparison - calculate fastest avg run throughput among baselines
        baseline_avg_runs = []
        for obj_id in baseline_objects:
            baseline_obj_data = df_filtered[df_filtered['object_id'] == obj_id]
            baseline_run_throughputs = baseline_obj_data[baseline_obj_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
            baseline_avg_run = baseline_run_throughputs.mean()
            baseline_avg_runs.append(baseline_avg_run)
        
        baseline_run_max_raw = max(baseline_avg_runs)
        
        httwo_run_speedup = httwo_avg_run_raw / baseline_run_max_raw
        htone_run_speedup = htone_avg_run_raw / baseline_run_max_raw
        
        self.add_result("httwo_run_speedup", round(httwo_run_speedup, 2))
        self.add_result("htone_run_speedup_percent", round(htone_run_speedup*100, 1))

        self.add_result("httwo_speedup_percent", round((httwo_fill_speedup + httwo_run_speedup) * 100 / 2 - 100, 1))
        
        # === Workload-specific analysis ===
        # Run A to Run B comparison (case 17->18)
        for obj_id, macro in object_to_macro.items():
            obj_data = df_filtered[df_filtered['object_id'] == obj_id]
            
            run_a_throughput_raw = obj_data[obj_data['case_id'] == 17]['run_throughput (ops/s)'].values[0]
            run_b_throughput_raw = obj_data[obj_data['case_id'] == 18]['run_throughput (ops/s)'].values[0]
            
            # Calculate percentage change using raw values
            if run_a_throughput_raw > 0:
                change_percent = ((run_b_throughput_raw - run_a_throughput_raw) / run_a_throughput_raw) * 100
                speedup_factor = run_b_throughput_raw / run_a_throughput_raw
                
                self.add_result(f"{macro}_a_to_b_speedup", round(speedup_factor, 1))
                self.add_result(f"{macro}_a_to_b_change_percent", round(abs(change_percent), 1))
        
        # === Positive vs Negative query analysis ===
        # Run C vs Run C- comparison (case 19 vs 22)
        for obj_id, macro in object_to_macro.items():
            obj_data = df_filtered[df_filtered['object_id'] == obj_id]
            
            run_c_pos_raw = obj_data[obj_data['case_id'] == 19]['run_throughput (ops/s)'].values[0]  # Positive
            run_c_neg_raw = obj_data[obj_data['case_id'] == 22]['run_throughput (ops/s)'].values[0]  # Negative
            
            # Calculate which is higher and by how much using raw values
            if run_c_neg_raw > run_c_pos_raw:
                speedup = run_c_neg_raw / run_c_pos_raw
                self.add_result(f"{macro}_neg_over_pos_speedup", round(speedup, 1))
            else:
                speedup = run_c_pos_raw / run_c_neg_raw
                self.add_result(f"{macro}_pos_over_neg_speedup", round(speedup, 1))
            
            # Also calculate percentage difference using raw values
            diff_percent = abs((run_c_neg_raw - run_c_pos_raw) / max(run_c_pos_raw, run_c_neg_raw)) * 100
            self.add_result(f"{macro}_pos_neg_diff_percent", round(diff_percent, 1))
        
        # === Junction's Run C speedup over Load ===
        junction_data = df_filtered[df_filtered['object_id'] == 15]
        junction_load_fill_raw = junction_data[junction_data['case_id'] == 17]['fill_throughput (ops/s)'].values[0]
        junction_run_c_raw = junction_data[junction_data['case_id'] == 19]['run_throughput (ops/s)'].values[0]
        
        junction_c_over_load_speedup = junction_run_c_raw / junction_load_fill_raw
        self.add_result("htfive_run_c_over_load_speedup", round(junction_c_over_load_speedup, 1))
        
        # === htfour workload variance analysis ===
        # Calculate coefficient of variation (CV) for htfour across all run workloads
        htfour_data = df_filtered[df_filtered['object_id'] == 7]
        htfour_run_throughputs = htfour_data[htfour_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
        htfour_mean = htfour_run_throughputs.mean()
        htfour_std = htfour_run_throughputs.std()
        htfour_cv_percent = (htfour_std / htfour_mean) * 100
        self.add_result("htfour_workload_variance_percent", round(htfour_cv_percent, 1))
    
    def calculate_tradeoff_metrics(self):
        """
        Calculate speed/space tradeoff metrics for the tradeoff.tex section.
        
        Uses: throughput_space_eff_results.csv
        Calculates: Space efficiency metrics and throughput analysis
        """
        df = self.get_dataframe('throughput_space_eff_results')
        
        # Object ID to macro mapping
        object_to_macro = {
            6: 'htthree',   # Cuckoo
            7: 'htfour',    # Iceberg
            15: 'htfive',   # Junction
            24: 'htsix',    # Baseline
            17: 'htone',    # TPHT
            23: 'httwo'     # Blast
        }

        baseline_objects = [6, 7, 15, 24]  # Cuckoo, Iceberg, Junction, htsix

        htone_space_eff_percent = 0
        httwo_space_eff_percent = 0

        baseline_space_eff_percent = []
        
        # === Maximum space efficiency analysis ===
        for obj_id, macro in object_to_macro.items():
            obj_data = df[df['object_id'] == obj_id]
            max_space_eff = obj_data['space_efficiency'].max()
            # Convert to percentage and round
            max_space_eff_percent = max_space_eff * 100

            if obj_id == 17:
                htone_space_eff_percent = max_space_eff_percent
            if obj_id == 23:
                httwo_space_eff_percent = max_space_eff_percent

            if obj_id in baseline_objects:
                baseline_space_eff_percent.append(max_space_eff_percent)

            self.add_result(f"{macro}_max_space_efficiency_percent", round(max_space_eff_percent, 1))
        
        mem_shave_percent = []

        for p in baseline_space_eff_percent:
            mem_shave_percent.append((1 - p / htone_space_eff_percent)*100)

        max_space_eff_percent = max(mem_shave_percent)
        self.add_result("htone_mem_shave_upper_percent", round(max_space_eff_percent, 1))

        min_space_eff_percent = min(mem_shave_percent)
        self.add_result("htone_mem_shave_lower_percent", round(min_space_eff_percent, 1))

        mean_space_eff_percent = np.mean(mem_shave_percent)
        self.add_result("htone_mem_shave_mean_percent", round(mean_space_eff_percent, 1))

        mem_shave_percent_httwo = []
        for p in baseline_space_eff_percent:
            mem_shave_percent_httwo.append((1 - p / httwo_space_eff_percent)*100)
        
        mean_space_eff_percent_httwo = np.mean(mem_shave_percent_httwo)
        self.add_result("httwo_mem_shave_mean_percent", round(mean_space_eff_percent_httwo, 1))




        # === Throughput drop analysis from low to high load factor ===
        # For positive query (case_id=9), analyze throughput drop from 0.05 to 0.7 load factor
        case9_data = df[df['case_id'] == 9]
        
        for obj_id, macro in object_to_macro.items():
            obj_case9 = case9_data[case9_data['object_id'] == obj_id]
            
            # Get throughput at 0.05 and 0.7 load factors
            low_lf_data = obj_case9[obj_case9['load_factor'] == 0.05]
            high_lf_data = obj_case9[obj_case9['load_factor'] == 0.7]
            
            if not low_lf_data.empty and not high_lf_data.empty:
                low_throughput = low_lf_data['throughput_millions'].iloc[0]
                high_throughput = high_lf_data['throughput_millions'].iloc[0]
                
                # Calculate percentage drop
                drop_percent = ((low_throughput - high_throughput) / low_throughput) * 100
                self.add_result(f"{macro}_throughput_drop_percent", round(drop_percent, 1))
            
            # Special case for htfour speedup near 0.5 load factor
            if obj_id == 7:  # htfour
                mid_lf_data_1 = obj_case9[obj_case9['load_factor'] == 0.55]
                mid_lf_data_2 = obj_case9[obj_case9['load_factor'] == 0.6]
                if not mid_lf_data_1.empty and not mid_lf_data_2.empty:
                    speedup_percent = ((mid_lf_data_2['throughput_millions'].iloc[0] - mid_lf_data_1['throughput_millions'].iloc[0]) / mid_lf_data_1['throughput_millions'].iloc[0]) * 100
                    self.add_result(f"{macro}_mid_load_speedup_percent", round(speedup_percent, 1))
        
        # === Performance comparison at 0.05 load factor ===
        # htfive vs httwo at 0.05 load factor for positive query (case_id=9)
        case9_data = df[df['case_id'] == 9]
        low_lf_case9 = case9_data[case9_data['load_factor'] == 0.05]
        
        htfive_low_throughput = low_lf_case9[low_lf_case9['object_id'] == 15]['throughput_millions'].iloc[0]
        httwo_low_throughput = low_lf_case9[low_lf_case9['object_id'] == 23]['throughput_millions'].iloc[0]
        
        throughput_ratio = htfive_low_throughput / httwo_low_throughput
        self.add_result("htfive_over_httwo_low_load_ratio", round(throughput_ratio, 1))

        # === httwo performance ratios at 50% and 70% space efficiency ===
        # Calculate for each case_id (positive query = case_id 9, insertion = case_id 1, negative query = case_id 10)
        case_ids = {
            'insertion': 1,
            'positive_query': 9,
            'negative_query': 10
        }

        for case_name, case_id in case_ids.items():
            # Get data for this case
            case_data = df[df['case_id'] == case_id]
            
            if not case_data.empty:
                # Find first data point at or above 0.50 and 0.70 space efficiency for each object
                space_eff_thresholds = [0.50, 0.70]
                results_by_threshold = {}
                
                for threshold in space_eff_thresholds:
                    results_by_threshold[threshold] = {}
                    
                    for obj_id in [6, 7, 15, 23, 24]:  # Include httwo (23) and all baselines
                        obj_case_data = case_data[case_data['object_id'] == obj_id].copy()
                        obj_case_data = obj_case_data.sort_values('space_efficiency')
                        
                        # Find first point at or above threshold
                        point = obj_case_data[obj_case_data['space_efficiency'] >= threshold]
                        if not point.empty:
                            results_by_threshold[threshold][obj_id] = {
                                'throughput': point.iloc[0]['throughput_millions'],
                                'space_eff': point.iloc[0]['space_efficiency']
                            }
                
                # Calculate ratios at 0.50
                if 0.50 in results_by_threshold and 23 in results_by_threshold[0.50]:
                    httwo_throughput_50 = results_by_threshold[0.50][23]['throughput']
                    baseline_objects = [obj_id for obj_id in [6, 7, 15, 24] if obj_id in results_by_threshold[0.50]]
                    
                    if baseline_objects:
                        baseline_throughputs = [results_by_threshold[0.50][obj_id]['throughput'] for obj_id in baseline_objects]
                        best_baseline_50 = max(baseline_throughputs)
                        ratio_50 = httwo_throughput_50 / best_baseline_50
                        
                        # Data 1: Percentage of fastest baseline (e.g., 0.9 -> 90%, 1.1 -> 110%)
                        ratio_50_percent = ratio_50 * 100
                        self.add_result(f"httwo_{case_name}_percent_of_fastest_50pct", round(ratio_50_percent, 1))
                        
                        # Data 2: Speed increase/decrease percent compared to fastest baseline
                        # If ratio is 1.1, it means httwo is 10% faster. If 0.9, it's 10% slower.
                        speed_diff_percent = abs((ratio_50 - 1) * 100)
                        self.add_result(f"httwo_{case_name}_speed_diff_50pct", round(speed_diff_percent, 1))
                    
    
    def calculate_load_factor_metrics(self):
        """
        Calculate load factor support metrics for the prob_analysis.tex section.
        
        Uses: load_factor_support_results.csv
        Calculates: Load factor support percentages for different bin sizes and workloads
        """
        df = self.get_dataframe('load_factor_support_results')
        
        # Filter for object_id=4 (our hash table implementation)
        df_filtered = df[df['object_id'] == 4].copy()
        
        # Get the maximum bin size (127, which is 2^7-1)
        max_bin_size = df_filtered['bin_size'].max()
        max_bin_data = df_filtered[df_filtered['bin_size'] == max_bin_size]
        
        if not max_bin_data.empty:
            # Load factor support for insertion-only workload (from CSV)
            insertion_only_support = max_bin_data['load_factor (%)'].iloc[0]
            self.add_result("insertion_only_load_factor_percent", round(insertion_only_support, 1))
            
            # Load factor support for deletion-included workload (hardcoded value of 95% from the plots)
            # This corresponds to the "With Deletion" line at bin size 127 in the plot
            deletion_included_support = 95.0
            self.add_result("deletion_included_load_factor_percent", round(deletion_included_support, 1))
    
    def calculate_occupancy_analysis_metrics(self):
        """
        Calculate occupancy analysis metrics for the prob_analysis.tex section.
        
        Uses: occupancy_experimental_box_random.csv (or other variants)
        Calculates: Percentages for empty bins, bins with one key, and whisker variance
        """
        # Try to get the random variant first, fall back to others if not available
        df = None
        variants = ['occupancy_experimental_box_random', 'occupancy_experimental_box_low_hamming', 
                   'occupancy_experimental_box_high_hamming', 'occupancy_experimental_box_sequential']
        
        for variant in variants:
            try:
                df = self.get_dataframe(variant)
                print(f"Using occupancy data from: {variant}")
                break
            except ValueError:
                continue
        
        if df is None:
            # Try the old naming convention as fallback
            try:
                df = self.get_dataframe('occupancy_experimental_box')
                print("Using occupancy data from: occupancy_experimental_box")
            except ValueError:
                available = ', '.join(self.dataframes.keys())
                raise ValueError(f"No occupancy experimental box data found. Available dataframes: {available}")
        
        # Get data for occupancy 0 (empty bins) and occupancy 1 (bins with one key)
        empty_bins_data = df[df['occupancy'] == 0]
        one_key_bins_data = df[df['occupancy'] == 1]
        
        if not empty_bins_data.empty and not one_key_bins_data.empty:
            # Convert to percentages and round to 2 decimal places
            empty_bins_percent = empty_bins_data['median'].iloc[0] * 100
            one_key_bins_percent = one_key_bins_data['median'].iloc[0] * 100
            
            self.add_result("empty_bins_percent", round(empty_bins_percent, 2))
            self.add_result("one_key_bins_percent", round(one_key_bins_percent, 2))
            
            # Calculate whisker variance as percentage difference from median
            # For empty bins (occupancy = 0)
            empty_median = empty_bins_data['median'].iloc[0]
            empty_upper_whisker = empty_bins_data['upper_whisker'].iloc[0]
            empty_lower_whisker = empty_bins_data['lower_whisker'].iloc[0]
            
            # Handle case where median might be very small (avoid division by very small numbers)
            if empty_median > 1e-10:  # Only calculate if median is not essentially zero
                empty_upper_diff_percent = abs((empty_upper_whisker - empty_median) / empty_median) * 100
                empty_lower_diff_percent = abs((empty_lower_whisker - empty_median) / empty_median) * 100
                empty_max_whisker_diff = max(empty_upper_diff_percent, empty_lower_diff_percent)
            else:
                empty_max_whisker_diff = 0.0
            
            # For bins with one key (occupancy = 1)
            one_median = one_key_bins_data['median'].iloc[0]
            one_upper_whisker = one_key_bins_data['upper_whisker'].iloc[0]
            one_lower_whisker = one_key_bins_data['lower_whisker'].iloc[0]
            
            # Handle case where median might be very small
            if one_median > 1e-10:  # Only calculate if median is not essentially zero
                one_upper_diff_percent = abs((one_upper_whisker - one_median) / one_median) * 100
                one_lower_diff_percent = abs((one_lower_whisker - one_median) / one_median) * 100
                one_max_whisker_diff = max(one_upper_diff_percent, one_lower_diff_percent)
            else:
                one_max_whisker_diff = 0.0
            
            # Take the maximum whisker difference across both occupancy levels
            max_whisker_diff = max(empty_max_whisker_diff, one_max_whisker_diff)
            
            self.add_result("max_whisker_diff_percent", round(max_whisker_diff, 3))
        else:
            print("Warning: Could not find data for occupancy 0 or 1 in the dataset")
    
    def calculate_tpht_thread_scaling_factor(self):
        df = self.get_dataframe('scaling_results')

        # Object IDs for TPHT variants
        htone_id = 17  # TPHT
        httwo_id = 20  # Blast
        
        # Case IDs to analyze
        target_cases = [1, 3, 6, 7]
        
        # Function to calculate scaling factor for a given object and case
        def calculate_scaling_factor(obj_id, case_id):
            case_data = df[(df['object_id'] == obj_id) & (df['case_id'] == case_id)]
            
            if case_data.empty:
                return None
            
            # Sort by thread number to ensure correct order
            case_data = case_data.sort_values('thread_num')
            
            # Restrict to specific thread counts for consistent scaling analysis
            target_threads = [1, 2, 4, 8, 16]  # Powers of 2 from 1 to 16
            
            # Filter data to only include target thread counts
            filtered_data = case_data[case_data['thread_num'].isin(target_threads)]
            
            if len(filtered_data) < 2:
                return None
            
            # Get thread counts and corresponding throughputs
            threads = filtered_data['thread_num'].tolist()
            throughputs = filtered_data['throughput (ops/s)'].tolist()
            
            # Use log-log linear regression to find power law relationship (like in coef.py)
            # throughput = a * threads^b, where b is the scaling exponent
            import math
            
            log_threads = [math.log(t) for t in threads]
            log_throughput = [math.log(tp) for tp in throughputs]
            
            n = len(threads)
            sum_log_threads = sum(log_threads)
            sum_log_throughput = sum(log_throughput)
            sum_log_threads_sq = sum(lt**2 for lt in log_threads)
            sum_log_products = sum(log_threads[i] * log_throughput[i] for i in range(n))
            
            # Calculate power law exponent: throughput = a * threads^b
            denominator = n * sum_log_threads_sq - sum_log_threads**2
            if denominator == 0:
                return None
                
            b = (n * sum_log_products - sum_log_threads * sum_log_throughput) / denominator
            
            # The scaling factor is the exponent b
            # Perfect linear scaling (y = kx) would have b = 1.0
            # b < 1.0 indicates diminishing returns with more threads
            # b > 1.0 indicates super-linear scaling (rare)
            return b
        
        # Calculate scaling factors for htone (object_id=17)
        htone_scaling_factors = []
        for case_id in target_cases:
            scaling_factor = calculate_scaling_factor(htone_id, case_id)
            if scaling_factor is not None:
                htone_scaling_factors.append(scaling_factor)
        
        # Calculate scaling factors for httwo (object_id=20)
        httwo_scaling_factors = []
        for case_id in target_cases:
            scaling_factor = calculate_scaling_factor(httwo_id, case_id)
            if scaling_factor is not None:
                httwo_scaling_factors.append(scaling_factor)
        
        # Calculate average scaling factors
        if htone_scaling_factors:
            htone_avg_scaling = sum(htone_scaling_factors) / len(htone_scaling_factors)
            self.add_result("htone_thread_scaling_factor", round(htone_avg_scaling, 3))
        
        if httwo_scaling_factors:
            httwo_avg_scaling = sum(httwo_scaling_factors) / len(httwo_scaling_factors)
            self.add_result("httwo_thread_scaling_factor", round(httwo_avg_scaling, 3))
        
        

    def calculate_incache_performance_degradation(self):

        df = self.get_dataframe('data_size_scaling_results')
        # Object IDs needed

        obj_ids = [17] # only for TPHT at the moment
        obj_names = ['htone']
        incache_table_size = [262143, 2097151]
        case_ids = [1] # only for insertion

        assert len(obj_ids) == len(obj_names)
        assert len(incache_table_size) == 2
        # filter out the throughput for the above three parameters
        # for each obj and case, calculate the throughput degradation
        # percentage corresponded to the two table sizes
        for obj_id in obj_ids:
            for case_id in case_ids:
                obj_case_data = df[(df['object_id'] == obj_id) & (df['case_id'] == case_id)]
                if obj_case_data.empty:
                    continue
                
                # Get throughput at the two incache table sizes
                throughputs = []
                for table_size in incache_table_size:
                    size_data = obj_case_data[obj_case_data['table_size'] == table_size]
                    if not size_data.empty:
                        throughputs.append(size_data['throughput (ops/s)'].values[0])
                
                if len(throughputs) == 2:
                    low_throughput, high_throughput = throughputs
                    degradation_percent = ((low_throughput - high_throughput) / low_throughput) * 100
                    obj_name = obj_names[obj_ids.index(obj_id)]
                    self.add_result(f"{obj_name}_insertion_incache_performance_degradation_percent", round(degradation_percent, 1))

    def calculate_inmem_performance_degradation(self):
        
        df = self.get_dataframe('data_size_scaling_results')
        # Object IDs needed

        obj_ids = [7] # only for Iceberg at the moment
        obj_name = ['htfour']
        inmem_table_size = [16777215, 134217727]
        case_ids = [1] # only for insertion

        assert len(obj_ids) == len(obj_name)
        assert len(inmem_table_size) == 2

        # filter out the throughput for the above three parameters
        # for each obj and case, calculate the throughput degradation
        # percentage corresponded to the two table sizes
        for obj_id in obj_ids:
            for case_id in case_ids:
                obj_case_data = df[(df['object_id'] == obj_id) & (df['case_id'] == case_id)]
                if obj_case_data.empty:
                    continue
                
                # Get throughput at the two inmem table sizes
                throughputs = []
                for table_size in inmem_table_size:
                    size_data = obj_case_data[obj_case_data['table_size'] == table_size]
                    if not size_data.empty:
                        throughputs.append(size_data['throughput (ops/s)'].values[0])
                
                if len(throughputs) == 2:
                    low_throughput, high_throughput = throughputs
                    degradation_percent = ((low_throughput - high_throughput) / low_throughput) * 100
                    obj_name = obj_name[obj_ids.index(obj_id)]
                    self.add_result(f"{obj_name}_insertion_inmem_performance_degradation_percent", round(degradation_percent, 1))

    def calculate_latency_shaving_metrics(self):
        """
        Calculate latency shaving metrics for the resizing.tex section.
        
        Uses: percentile_results.csv
        Calculates: Percentage reduction in maximum tail latency for httwo vs fastest baseline
        """
        df = self.get_dataframe('percentile_results')
        
        # Filter for positive query operations (operation_type=0) and maximum percentile (100.0)
        max_latency_data = df[(df['operation_type'] == 0) & (df['percentile'] == 100.0)]
        
        # Object ID mappings
        baseline_objects = [6, 7, 15, 24]  # htthree, htfour, htfive, htsix
        httwo_object_id = 21
        
        # Get httwo maximum latency
        httwo_max_data = max_latency_data[max_latency_data['object_id'] == httwo_object_id]
        if httwo_max_data.empty:
            print(f"Warning: No data found for httwo (object_id={httwo_object_id})")
            return
        
        httwo_max_latency = httwo_max_data['latency_ns'].iloc[0]
        
        # Get baseline maximum latencies and find the fastest (lowest latency)
        baseline_max_latencies = []
        for obj_id in baseline_objects:
            baseline_data = max_latency_data[max_latency_data['object_id'] == obj_id]
            if not baseline_data.empty:
                baseline_max_latencies.append(baseline_data['latency_ns'].iloc[0])
        
        if not baseline_max_latencies:
            print("Warning: No baseline data found")
            return
        
        # Find the fastest baseline (minimum latency)
        fastest_baseline_latency = min(baseline_max_latencies)
        
        # Calculate percentage reduction
        if fastest_baseline_latency > 0:
            print(f"Fastest baseline latency: {fastest_baseline_latency}, httwo max latency: {httwo_max_latency}")
            latency_reduction_percent = ((fastest_baseline_latency - httwo_max_latency) / fastest_baseline_latency) * 100
            self.add_result("httwo_latency_shave_percent", round(latency_reduction_percent, 1))
        else:
            print("Warning: Invalid baseline latency data")
    
    def calculate_resizing_metrics(self):
        """
        Calculate resizing throughput metrics for the resizing.tex section.
        
        Uses: ycsb_results.csv (entry_id=1 for resizing experiments)
        Calculates: Average throughputs and performance analysis with resizing enabled
        """
        df = self.get_dataframe('ycsb_results')
        
        # Filter for entry_id=1 (resizing experiments) 
        df_resizing = df[df['entry_id'] == 1].copy()
        
        # Also get non-resizing data (entry_id=0) for comparison
        df_no_resizing = df[df['entry_id'] == 0].copy()
        
        # Object ID to macro mapping (note: httwo is object_id=21 in resizing, 18 in resizing)
        resizing_object_to_macro = {
            6: 'htthree',   # Cuckoo
            7: 'htfour',    # Iceberg  
            15: 'htfive',   # Junction
            24: 'htsix',    # Baseline
            18: 'htone',    # TPHT (object_id=18 in resizing experiments)
            21: 'httwo'     # Blast (object_id=21 in resizing experiments)
        }
        
        # === Average throughput calculations for resizing ===
        # Calculate for all hash tables in resizing experiments
        for obj_id, macro in resizing_object_to_macro.items():
            ht_resizing_data = df_resizing[df_resizing['object_id'] == obj_id]
            
            if not ht_resizing_data.empty:
                # Average fill throughput for Load phase (case_id=17)
                fill_data = ht_resizing_data[ht_resizing_data['case_id'] == 17]['fill_throughput (ops/s)']
                if not fill_data.empty:
                    ht_resizing_fill_raw = fill_data.values[0]
                    self.add_result(f"{macro}_resizing_avg_fill_throughput", round(ht_resizing_fill_raw / 1_000_000, 1))
                
                # Average run throughput across all run phases (case_ids 17-22)
                ht_resizing_run_throughputs_raw = ht_resizing_data[ht_resizing_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
                if not ht_resizing_run_throughputs_raw.empty:
                    ht_resizing_avg_run_raw = ht_resizing_run_throughputs_raw.mean()
                    self.add_result(f"{macro}_resizing_avg_run_throughput", round(ht_resizing_avg_run_raw / 1_000_000, 1))
        
        # === Throughput decrease analysis ===
        # Compare resizing vs non-resizing for htfour to calculate throughput decrease
        
        # Get htfour non-resizing average (across fill and run)
        htfour_no_resizing_data = df_no_resizing[df_no_resizing['object_id'] == 7]
        htfour_no_resizing_fill_raw = htfour_no_resizing_data[htfour_no_resizing_data['case_id'] == 17]['fill_throughput (ops/s)'].values[0]
        htfour_no_resizing_run_throughputs_raw = htfour_no_resizing_data[htfour_no_resizing_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
        htfour_no_resizing_avg_run_raw = htfour_no_resizing_run_throughputs_raw.mean()
        htfour_no_resizing_combined_avg = (htfour_no_resizing_fill_raw + htfour_no_resizing_avg_run_raw) / 2
        
        # Get htfour resizing average (across fill and run)
        htfour_resizing_data = df_resizing[df_resizing['object_id'] == 7]
        htfour_resizing_fill_raw = htfour_resizing_data[htfour_resizing_data['case_id'] == 17]['fill_throughput (ops/s)'].values[0]
        htfour_resizing_run_throughputs_raw = htfour_resizing_data[htfour_resizing_data['case_id'].isin([17, 18, 19, 20, 21, 22])]['run_throughput (ops/s)']
        htfour_resizing_avg_run_raw = htfour_resizing_run_throughputs_raw.mean()
        htfour_resizing_combined_avg = (htfour_resizing_fill_raw + htfour_resizing_avg_run_raw) / 2
        
        # Calculate percentage decrease
        htfour_throughput_decrease_percent = ((htfour_no_resizing_combined_avg - htfour_resizing_combined_avg) / htfour_no_resizing_combined_avg) * 100
        self.add_result("htfour_resizing_throughput_decrease_percent", round(htfour_throughput_decrease_percent, 1))
    
    def calculate_resizing_space_efficiency_metrics(self):
        """
        Calculate worst-case space efficiency metrics for resizing RSS experiment.
        
        Uses: resizing_rss.csv
        Calculates: Worst space efficiency for each method (object_id) using formula:
        space_efficiency = 80530636 * 16 * completion / memory_mb / 100 / 2^20
        Only considers points with completion > 25
        """
        df = self.get_dataframe('resizing_rss')
        
        # Object ID to macro mapping (based on the existing mappings in the file)
        object_to_macro = {
            6: 'htthree',   # Cuckoo
            7: 'htfour',    # Iceberg
            15: 'htfive',   # Junction
            18: 'htone',    # TPHT
            21: 'httwo',    # Blast
            24: 'htsix',     # Baseline
            25: 'htseven',     # Stagger ByteArray Chained HT
        }
        
        # Filter for completion > 25
        df_filtered = df[df['completion'] > 25].copy()
        
        if df_filtered.empty:
            print("Warning: No data points with completion > 25 found")
            return
        
        # Calculate space efficiency for each row
        # Formula: 80530636 * 16 * completion / memory_mb / 100 / 2^20
        df_filtered['space_efficiency'] = (80530636 * 16 * df_filtered['completion'] / 
                                      df_filtered['memory_mb'] / 100 / (2**20))
        
        # Find worst (minimum) space efficiency for each object_id
        for obj_id, macro in object_to_macro.items():
            obj_data = df_filtered[df_filtered['object_id'] == obj_id]
            
            if not obj_data.empty:
                worst_space_efficiency = obj_data['space_efficiency'].min()
                # Convert to percentage
                worst_space_efficiency_percent = worst_space_efficiency * 100
                
                self.add_result(f"{macro}_worst_resizing_space_efficiency_percent", 
                              round(worst_space_efficiency_percent, 1))
                
                print(f"{macro} (object_id={obj_id}): worst space efficiency = {worst_space_efficiency_percent:.1f}%")
            else:
                print(f"Warning: No data found for {macro} (object_id={obj_id}) with completion > 25")
    
    # ==========================================================================
    # ADD YOUR CALCULATION FUNCTIONS HERE
    # ==========================================================================
    
    def your_calculation_function_template(self):
        """
        Template for adding new calculation functions.
        
        Uses: [specify which CSV file(s)]
        Calculates: [describe what this function calculates]
        """
        # Example:
        # df = self.get_dataframe('your_csv_name')
        # result = df['column'].mean()  # or any other calculation
        # self.add_result("your_metric_name", result)
        pass
    
    # ==========================================================================
    # MAIN EXECUTION FUNCTIONS
    # ==========================================================================
    
    def run_all_calculations(self):
        """Run all calculation functions."""
        print("Running all calculations...")
        
        # Add calls to all your calculation functions here
        self.calculate_throughput_metrics()
        self.calculate_tradeoff_metrics()
        self.calculate_load_factor_metrics()
        self.calculate_occupancy_analysis_metrics()
        self.calculate_tpht_thread_scaling_factor()
        self.calculate_incache_performance_degradation()
        self.calculate_inmem_performance_degradation()
        self.calculate_latency_shaving_metrics()
        self.calculate_resizing_metrics()
        self.calculate_resizing_space_efficiency_metrics()
        # Add calls to your new functions here:
        # self.your_new_function()
        
        print(f"Generated {len(self.results)} results")
    
    def make_latex_safe_name(self, name: str) -> str:
        """Convert a name to be LaTeX-safe for use in newcommand."""
        # Keep only the base name but convert numbers to words
        number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        
        # Replace underscores with nothing (keep original name structure)
        latex_name = name.replace("_", "")
        
        # Replace numbers with words
        for digit, word in number_words.items():
            latex_name = latex_name.replace(digit, word)
        
        # Remove other invalid characters
        latex_name = latex_name.replace("-", "")
        latex_name = latex_name.replace(".", "")
        latex_name = latex_name.replace(" ", "")
        
        # Ensure it starts with a letter (should be fine now since we convert numbers)
        if latex_name and not latex_name[0].isalpha():
            latex_name = "data" + latex_name
        
        return latex_name
    
    def save_results(self):
        """Save results to CSV file."""
        if not self.results:
            print("No results to save!")
            return
        
        # Create output dataframe
        results_df = pd.DataFrame(self.results, columns=['id', 'name', 'value'])
        
        # Save to CSV
        results_df.to_csv(self.output_file, index=False)
        print(f"Results saved to {self.output_file}")
        
        # Display results for verification
        print("\nGenerated results:")
        print(results_df.to_string(index=False))
    
    def save_latex_commands(self):
        """Save results as LaTeX newcommand definitions."""
        if not self.results:
            print("No results to save as LaTeX!")
            return
        
        latex_lines = []
        
        for _, name, value in self.results:
            # Create LaTeX-safe command name
            latex_name = self.make_latex_safe_name(name)
            
            # Format value appropriately with 4 significant digits
            if isinstance(value, (int, float)):
                # Format all numbers to 4 significant digits
                formatted_value = f"{value:.4g}"
            else:
                formatted_value = str(value)
            
            # Create newcommand line
            latex_line = f"\\newcommand{{\\{latex_name}}}{{{formatted_value}}}"
            latex_lines.append(latex_line)
        
        # Join with empty lines between commands
        latex_content = "\n\n".join(latex_lines)
        
        # Save to file
        with open(self.latex_file, 'w') as f:
            f.write(latex_content)
        
        print(f"LaTeX commands saved to {self.latex_file}")
        print("\nGenerated LaTeX commands:")
        print(latex_content)
    
    def show_available_data(self):
        """Display information about available datasets."""
        print("Available datasets:")
        print("=" * 50)
        for name, df in self.dataframes.items():
            print(f"{name}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            if not df.empty:
                print(f"  Sample data:")
                print(f"    {df.head(2).to_string(index=False)}")
            print()

def main():
    """Main execution function."""
    processor = DataProcessor()
    
    # Show available data for reference
    processor.show_available_data()
    
    # Run all calculations
    processor.run_all_calculations()
    
    # Save results
    processor.save_results()
    
    # Save LaTeX commands
    processor.save_latex_commands()

if __name__ == "__main__":
    main()