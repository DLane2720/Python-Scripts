#!/usr/bin/env python3

"""
ROS Bag Analyzer - Message Size and Topic Analysis Tool

This script analyzes ROS1 bag files to collect statistics about message sizes and topic frequencies.
It processes multiple bag files listed in a CSV and generates detailed reports about message counts,
sizes, and topic distributions.

Key features:
- Analyzes message sizes for specified target topics
- Tracks message counts across all topics
- Generates summary statistics
- Supports graceful exit handling
- Saves progress incrementally

Dependencies:
    - rosbag
    - pandas
    - tqdm
"""

import rosbag
import os
import pandas as pd
from tqdm import tqdm
import argparse
from typing import Dict, Any, List, Tuple
import sys
from datetime import datetime
from collections import defaultdict
from io import StringIO, BytesIO
import signal
import time

class GracefulExitHandler:
    """
    Handles graceful termination of the script on SIGINT/SIGTERM signals.
    Allows the current bag processing to complete before exiting.
    """
    def __init__(self):
        self.exit_now = False
        # Register signal handlers for clean termination
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        """Signal handler that sets the exit flag when termination is requested."""
        print("\nReceived signal to terminate. Will exit after current bag finishes processing...")
        self.exit_now = True

def get_message_sizes(bag_path: str, topic_name: str) -> List[Dict[str, Any]]:
    """
    Get the serialized size of each message for a given topic in a bag file.
    
    Args:
        bag_path: Path to the ROS bag file
        topic_name: Name of the topic to analyze
        
    Returns:
        List of dictionaries containing message sizes and metadata:
        [
            {
                'bag_file': str,          # Name of the bag file
                'topic': str,             # Topic name
                'timestamp': float,       # Message timestamp
                'size': int              # Serialized message size in bytes
            },
            ...
        ]
    """
    message_data = []
    
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            # Iterate through messages and calculate their serialized sizes
            for _, msg, t in bag.read_messages(topics=[topic_name]):
                buffer = BytesIO()
                msg.serialize(buffer)
                message_size = len(buffer.getvalue())
                message_data.append({
                    'bag_file': os.path.basename(bag_path),
                    'topic': topic_name,
                    'timestamp': t.to_sec(),
                    'size': message_size
                })
    except Exception as e:
        print(f"Error getting message sizes for topic {topic_name}: {str(e)}")
        
    return message_data

def analyze_bag(bag_path: str, target_topics: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Analyze a ROS1 bag file for all topics and perform detailed analysis of target topics.
    
    Args:
        bag_path: Path to the ROS bag file
        target_topics: List of topics for detailed size analysis
        
    Returns:
        Tuple containing:
        - Dictionary with bag statistics including counts and size metrics
        - List of individual message size data for target topics
    """
    # Initialize statistics dictionary with basic metadata
    stats = {
        'filepath': bag_path,
        'filename': os.path.basename(bag_path),
        'status': 'success',
        'error_msg': '',
        'topic_counts': defaultdict(int),
    }
    
    all_message_sizes = []
    
    # Initialize statistics fields for each target topic
    for topic in target_topics:
        stats[f'{topic}_count'] = 0
        stats[f'{topic}_total_size'] = 0
        stats[f'{topic}_avg_size'] = 0
        stats[f'{topic}_min_size'] = 0
        stats[f'{topic}_max_size'] = 0
    
    try:
        # First pass: Count messages for all topics
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, _, _ in bag.read_messages():
                stats['topic_counts'][topic] += 1
        
        # Second pass: Detailed size analysis for target topics
        for topic in target_topics:
            message_data = get_message_sizes(bag_path, topic)
            all_message_sizes.extend(message_data)
            
            # Calculate statistics if messages were found
            if message_data:
                sizes = [d['size'] for d in message_data]
                stats[f'{topic}_count'] = len(sizes)
                stats[f'{topic}_total_size'] = sum(sizes)
                stats[f'{topic}_avg_size'] = sum(sizes) / len(sizes)
                stats[f'{topic}_min_size'] = min(sizes)
                stats[f'{topic}_max_size'] = max(sizes)
                
    except Exception as e:
        # Mark bag as failed if any error occurs
        stats['status'] = 'failed'
        stats['error_msg'] = str(e)
        stats['topic_counts'] = defaultdict(int)
    
    return stats, all_message_sizes

def save_results(stats_list: List[Dict], all_topics: set, target_topics: List[str], 
                processed_bags: int, failed_bags: int, output_file: str):
    """
    Save analysis results to CSV file using efficient DataFrame construction.
    
    Args:
        stats_list: List of statistics dictionaries from analyzed bags
        all_topics: Set of all unique topics found across bags
        target_topics: List of topics that received detailed analysis
        processed_bags: Count of successfully processed bags
        failed_bags: Count of bags that failed processing
        output_file: Path to save the CSV results
    """
    # Pre-allocate data dictionary for DataFrame columns
    data = {
        'filepath': [],
        'filename': [],
        'status': [],
        'error_msg': []
    }
    
    # Populate base columns
    for s in stats_list:
        data['filepath'].append(s.get('filepath', ''))
        data['filename'].append(s.get('filename', ''))
        data['status'].append(s.get('status', ''))
        data['error_msg'].append(s.get('error_msg', ''))
    
    # Add detailed statistics for target topics
    for topic in target_topics:
        data[f'{topic}_count'] = [s.get(f'{topic}_count', 0) for s in stats_list]
        data[f'{topic}_total_size'] = [s.get(f'{topic}_total_size', 0) for s in stats_list]
        data[f'{topic}_avg_size'] = [s.get(f'{topic}_avg_size', 0) for s in stats_list]
        data[f'{topic}_min_size'] = [s.get(f'{topic}_min_size', 0) for s in stats_list]
        data[f'{topic}_max_size'] = [s.get(f'{topic}_max_size', 0) for s in stats_list]
    
    # Add message counts for all topics
    for topic in sorted(all_topics):
        data[f'topic_{topic}'] = [s['topic_counts'].get(topic, 0) for s in stats_list]
    
    # Create DataFrame and calculate summary statistics
    results_df = pd.DataFrame(data)
    
    # Generate summary row
    summary = {
        'filepath': 'SUMMARY',
        'filename': 'SUMMARY',
        'status': f'Processed: {processed_bags}, Failed: {failed_bags}',
        'error_msg': ''
    }
    
    # Calculate summary statistics for target topics
    for topic in target_topics:
        total_count = sum(s.get(f'{topic}_count', 0) for s in stats_list)
        total_size = sum(s.get(f'{topic}_total_size', 0) for s in stats_list)
        
        summary[f'{topic}_count'] = total_count
        summary[f'{topic}_total_size'] = total_size
        summary[f'{topic}_avg_size'] = total_size / total_count if total_count > 0 else 0
        
        # Calculate global min/max across all bags
        min_sizes = [s.get(f'{topic}_min_size', float('inf')) for s in stats_list if s.get(f'{topic}_min_size', 0) > 0]
        max_sizes = [s.get(f'{topic}_max_size', 0) for s in stats_list if s.get(f'{topic}_max_size', 0) > 0]
        summary[f'{topic}_min_size'] = min(min_sizes) if min_sizes else 0
        summary[f'{topic}_max_size'] = max(max_sizes) if max_sizes else 0
    
    # Add topic counts to summary
    for topic in all_topics:
        summary[f'topic_{topic}'] = sum(s['topic_counts'].get(topic, 0) for s in stats_list)
    
    # Append summary row and save
    results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)
    results_df.to_csv(output_file, index=False)

def append_message_sizes(message_sizes: List[Dict], output_file: str):
    """
    Append new message size data to CSV file or create new file if none exists.
    
    Args:
        message_sizes: List of dictionaries containing message size data
        output_file: Path to the CSV file for storing message sizes
    """
    if not message_sizes:
        return
        
    new_df = pd.DataFrame(message_sizes)
    
    if os.path.exists(output_file):
        # Append to existing file without headers
        new_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # Create new file with headers
        new_df.to_csv(output_file, index=False)

def main():
    """
    Main entry point for the ROS bag analyzer script.
    Handles argument parsing and orchestrates the analysis process.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Analyze ROS1 bag topics and message sizes')
    parser.add_argument('--csv', default='bag_files_macOS.csv',
                      help='CSV file containing bag file paths (default: bag_files_macOS.csv)')
    parser.add_argument('--output', default='01ros_msg_results_final.csv',
                      help='Output CSV file (default: ros_msg_results.csv)')
    parser.add_argument('--sizes-output', default='01msg_sizes_final.csv',
                      help='Output CSV file for individual message sizes (default: msg_sizes.csv)')
    
    args = parser.parse_args()
    
    # Initialize graceful exit handler for clean termination
    exit_handler = GracefulExitHandler()
    
    # Define target topics for detailed analysis
    target_topics = [
        '/minion/vision/objects_w_class',
        '/minion/mapper/objects_w_class',
        '/minion/PCM/livox/objects_w_class'
    ]
    
    # Verify input CSV exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' does not exist!")
        sys.exit(1)
    
    # Read bag file paths from CSV
    try:
        df = pd.read_csv(args.csv)
        bag_files = [os.path.join(row['filepath'], row['filename']) for _, row in df.iterrows()]
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        sys.exit(1)
    
    print(f"Found {len(bag_files)} bag files in CSV")
    
    # Initialize tracking variables
    all_stats = []
    all_topics = set()
    processed_bags = 0
    failed_bags = 0
    
    # Process each bag file with progress bar
    for bag_path in tqdm(bag_files, desc="Processing bags"):
        try:
            # Analyze current bag
            stats, message_sizes = analyze_bag(bag_path, target_topics)
            all_stats.append(stats)
            
            # Update statistics
            if stats['status'] == 'success':
                processed_bags += 1
                all_topics.update(stats['topic_counts'].keys())
            else:
                failed_bags += 1
            
            # Save incremental progress
            save_results(all_stats, all_topics, target_topics, processed_bags, failed_bags, args.output)
            append_message_sizes(message_sizes, args.sizes_output)
            
        except Exception as e:
            print(f"\nError processing {bag_path}: {str(e)}")
            failed_bags += 1
        
        # Check for exit request
        if exit_handler.exit_now:
            print("\nExiting early due to user request...")
            break
    
    # Print final analysis summary
    print("\nAnalysis Complete!")
    print(f"Successfully processed {processed_bags}/{len(bag_files)} bags")
    print(f"Found {len(all_topics)} unique topics across all bags")
    
    # Calculate and display final statistics for target topics
    for topic in target_topics:
        total_count = sum(s.get(f'{topic}_count', 0) for s in all_stats)
        total_size = sum(s.get(f'{topic}_total_size', 0) for s in all_stats)
        
        if total_count > 0:
            avg_size = total_size / total_count
            print(f"\nResults for topic '{topic}':")
            print(f"Total messages: {total_count:,}")
            print(f"Average message size: {avg_size:.2f} bytes")
            print(f"Total data size: {total_size/1024/1024:.2f} MB")
            
            # Calculate global min/max sizes
            min_sizes = [s.get(f'{topic}_min_size', float('inf')) for s in all_stats if s.get(f'{topic}_min_size', 0) > 0]
            max_sizes = [s.get(f'{topic}_max_size', 0) for s in all_stats if s.get(f'{topic}_max_size', 0) > 0]
            if min_sizes and max_sizes:
                print(f"Min message size: {min(min_sizes):,} bytes")
                print(f"Max message size: {max(max_sizes):,} bytes")
    
    if failed_bags > 0:
        print(f"\nFailed to process {failed_bags} bags. Check {args.output} for details.")

if __name__ == '__main__':
    main()