#!/usr/bin/env python3

"""
Script to stop and remove Docker containers matching specific patterns:
- Container names containing "faiss" or "ann"
- Containers created by command "python3 -u run_algo*"
Stops when no matching containers are found in the last 5 consecutive runs
"""

import subprocess
import sys
import time
from collections import deque


def get_all_containers():
    """Get all containers with ID, name, and command."""
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--format', '{{.ID}}|{{.Names}}|{{.Command}}'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split('|', 2)
                if len(parts) == 3:
                    containers.append({
                        'id': parts[0],
                        'name': parts[1],
                        'command': parts[2]
                    })
        return containers
    except Exception as e:
        print(f"Error getting containers: {e}", file=sys.stderr)
        return []


def should_match_container(container):
    """Check if container matches the criteria."""
    name = container['name']
    command = container['command']
    
    # Check if name matches "*faiss*" or "*ann*"
    if 'faiss' in name or 'ann' in name or 'milvus' in name:
        return True
    
    # Check if command matches "python3 -u run_algo*"
    print(name)
    print(command)
    if 'python3 -u run_algo…' in command:
        print("command matches")
        return True
    
    return False


def stop_and_remove_containers():
    """Stop and remove containers matching the criteria."""
    all_containers = get_all_containers()
    
    if not all_containers:
        print("No containers found", file=sys.stderr)
        return 0
    
    # Filter containers based on criteria
    matching_containers = [
        c for c in all_containers if should_match_container(c)
    ]
    
    if not matching_containers:
        print("No matching containers found", file=sys.stderr)
        return 0
    
    count_before = len(matching_containers)
    container_ids = [c['id'] for c in matching_containers]
    
    # Stop matching containers
    print(f"Stopping {count_before} matching container(s)...", file=sys.stderr)
    subprocess.run(
        ['docker', 'stop'] + container_ids,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait a moment for containers to stop
    time.sleep(1)
    
    # Remove matching containers
    print(f"Removing {count_before} matching container(s)...", file=sys.stderr)
    subprocess.run(
        ['docker', 'rm'] + container_ids,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    return count_before


def main():
    """Main loop."""
    print("Starting Docker container cleanup...")
    
    # Track container counts from last 5 runs
    counts = deque(maxlen=5)
    run = 1
    
    while True:
        print()
        print(f"=== Run {run} ===")
        
        # Stop and remove containers, get count
        count = stop_and_remove_containers()
        
        # Add count to deque (automatically keeps only last 5)
        counts.append(count)
        
        print(f"Containers processed in this run: {count}")
        print(f"Last 5 runs: {list(counts)}")
        
        # Check if last 5 runs all had 0 containers
        if len(counts) == 5 and all(c == 0 for c in counts):
            print()
            print("No containers found in last 5 consecutive runs. Exiting.")
            break
        
        # Wait 3 seconds before next attempt
        print("Waiting 3 seconds before next attempt...")
        time.sleep(3)
        
        run += 1
    
    print("Cleanup complete!")


if __name__ == '__main__':
    main()

