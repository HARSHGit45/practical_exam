import heapq

def calculate_lower_bound(cost_matrix, assigned_workers, current_cost):
    n = len(cost_matrix)
    unassigned_workers = set(range(n)) - assigned_workers
    min_additional_cost = 0
    
    for job in range(n):
        if job not in assigned_workers:
            min_cost = float('inf')
            for worker in unassigned_workers:
                min_cost = min(min_cost, cost_matrix[worker][job])
            min_additional_cost += min_cost
    
    return current_cost + min_additional_cost

def branch_and_bound_assignment(cost_matrix):
    n = len(cost_matrix)
    pq = []  # Priority queue for states (min-heap)
    
    # Initial state: no jobs assigned, cost is 0
    initial_state = (0, 0, set(), [])
    heapq.heappush(pq, initial_state)
    
    best_cost = float('inf')
    best_assignment = []
    
    while pq:
        lb, cost, assigned_workers, assignment = heapq.heappop(pq)
        
        # If we have assigned all jobs, check for optimality
        if len(assignment) == n:
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
            continue
        
        # Branching: Assign next job to an unassigned worker
        next_job = len(assignment)
        for worker in range(n):
            if worker not in assigned_workers:
                new_cost = cost + cost_matrix[worker][next_job]
                new_assigned_workers = assigned_workers | {worker}
                new_assignment = assignment + [(worker, next_job)]
                lb = calculate_lower_bound(cost_matrix, new_assigned_workers, new_cost)
                
                if lb < best_cost:  # Prune branches that cannot improve the best cost
                    heapq.heappush(pq, (lb, new_cost, new_assigned_workers, new_assignment))
    
    return best_cost, best_assignment

# Example usage
cost_matrix = [
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4]
]

best_cost, best_assignment = branch_and_bound_assignment(cost_matrix)
print("Minimum cost:", best_cost)
print("Best assignment (worker, job):", best_assignment)
