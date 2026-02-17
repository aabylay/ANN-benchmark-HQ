#!/bin/bash

# Script to stop and remove Docker containers matching specific patterns:
# - Container names containing "faiss" or "ann"
# - Containers created by command "python3 -u run_algo*"
# Stops when no matching containers are found in the last 5 consecutive runs

# Array to track container counts from last 5 runs
declare -a counts=()

# Function to stop and remove containers
stop_and_remove_containers() {
    # Get list of all containers with ID, name, and command
    local all_containers=$(docker ps -a --format "{{.ID}}|{{.Names}}|{{.Command}}" 2>/dev/null)
    
    if [ -z "$all_containers" ]; then
        echo "No containers found" >&2
        echo "0"
        return
    fi
    
    # Filter containers based on criteria:
    # 1. Name matches "*faiss*" or "*ann*"
    # 2. Command matches "python3 -u run_algo*"
    local matching_containers=""
    local count_before=0
    
    while IFS='|' read -r container_id container_name container_command; do
        local should_match=false
        
        # Check if name matches "*faiss*" or "*ann*"
        if [[ "$container_name" == *"faiss"* ]] || [[ "$container_name" == *"ann"* ]]; then
            should_match=true
        fi
        
        # Check if command matches "python3 -u run_algo*"
        if [[ "$container_command" == "python3 -u run_algo"* ]]; then
            should_match=true
        fi
        
        if [ "$should_match" = true ]; then
            matching_containers="$matching_containers $container_id"
            ((count_before++))
        fi
    done <<< "$all_containers"
    
    if [ $count_before -eq 0 ]; then
        echo "No matching containers found" >&2
        echo "0"
        return
    fi
    
    # Trim leading space
    matching_containers=$(echo "$matching_containers" | sed 's/^ *//')
    
    # Stop matching containers
    echo "Stopping $count_before matching container(s)..." >&2
    docker stop $matching_containers >/dev/null 2>&1
    
    # Wait a moment for containers to stop
    sleep 1
    
    # Remove matching containers
    echo "Removing $count_before matching container(s)..." >&2
    docker rm $matching_containers >/dev/null 2>&1
    
    # Return count of containers processed (only this goes to stdout)
    echo "$count_before"
}

# Main loop
echo "Starting Docker container cleanup..."
run=1

while true; do
    echo ""
    echo "=== Run $run ==="
    
    # Stop and remove containers, get count
    count=$(stop_and_remove_containers)
    
    # Ensure count is numeric (strip any whitespace)
    count=$(echo "$count" | tr -d '[:space:]')
    
    # Add count to array (keep only last 5)
    counts+=($count)
    if [ ${#counts[@]} -gt 5 ]; then
        counts=("${counts[@]:1}")  # Remove first element
    fi
    
    echo "Containers processed in this run: $count"
    echo "Last 5 runs: ${counts[@]}"
    
    # Check if last 5 runs all had 0 containers
    if [ ${#counts[@]} -eq 5 ]; then
        all_zero=true
        for c in "${counts[@]}"; do
            # Ensure we're comparing numbers
            c=$(echo "$c" | tr -d '[:space:]')
            if [ "$c" -ne 0 ] 2>/dev/null; then
                all_zero=false
                break
            fi
        done
        
        if [ "$all_zero" = true ]; then
            echo ""
            echo "No containers found in last 5 consecutive runs. Exiting."
            break
        fi
    fi
    
    # Wait 3 seconds before next attempt
    echo "Waiting 3 seconds before next attempt..."
    sleep 3
    
    ((run++))
done

echo "Cleanup complete!"

