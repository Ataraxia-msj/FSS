import numpy as np

class SemiconductorEnv:
    def __init__(self, problem_data):
        """
        Initialize the environment with problem-specific data.
        
        Args:
            problem_data (dict): Contains N_M, N_O, N_J, job_types, processing_times,
                                 setup_times, alternative_machines, etc.
        """
        # Extract problem data
        self.N_M = problem_data['N_M']  # Number of machines
        self.N_O = problem_data['N_O']  # Number of operation types
        self.N_J = problem_data['N_J']  # Number of job types
        self.job_types = problem_data['job_types']  # List of operation type sequences per job type
        self.processing_times = problem_data['processing_times']  # p_{i,j} for O_{i,j}
        self.setup_times = problem_data['setup_times']  # sigma_{i',j',i,j}
        self.alternative_machines = problem_data['alternative_machines']  # E_{i,j}
        self.production_requirements = problem_data['production_requirements']  # P(J_i)

        # Precompute min and max values for action normalization
        self.min_setup_time = 0
        self.max_setup_time = np.max(self.setup_times)
        self.min_processing_time = np.min(self.processing_times)
        self.max_processing_time = np.max(self.processing_times)
        self.max_ops_per_job = max(len(seq) for seq in self.job_types)
        self.min_remaining_ops = 0
        self.max_remaining_ops = self.max_ops_per_job - 1
        self.max_remaining_time = max(
            sum(self.processing_times[i][j] for j in range(len(seq)))
            for i, seq in enumerate(self.job_types)
        )
        self.min_remaining_time = 0

        # Initialize internal state
        self.current_time = 0.0
        self.machines = []  # List of (status, setup_type, finish_time, operation_idx)
        self.operations = []  # List of (job_idx, op_position, status)
        self.jobs = []  # List of (job_type_idx, operation_count)
        self._initialize_problem()

    def _initialize_problem(self):
        """Set up jobs, operations, and machines at the start."""
        # Initialize machines: all idle, random initial setup
        self.machines = [
            ('idle', np.random.randint(self.N_O), 0.0, -1)
            for _ in range(self.N_M)
        ]
        
        # Create jobs and operations based on production requirements
        job_idx = 0
        for job_type_idx, count in enumerate(self.production_requirements):
            for _ in range(count):
                self.jobs.append((job_type_idx, len(self.job_types[job_type_idx])))
                # Add operations for this job
                for pos, op_type in enumerate(self.job_types[job_type_idx]):
                    status = 'waiting' if pos == 0 else 'not_ready'
                    self.operations.append((job_idx, pos, status))
                job_idx += 1

    def reset(self):
        """Reset the environment to initial state."""
        self.current_time = 0.0
        self.machines = []
        self.operations = []
        self.jobs = []
        self._initialize_problem()
        return self.get_state()

    def get_state(self):
        """
        Compute the current state as a 3 x N_O vector.
        
        Returns:
            np.ndarray: Concatenated state vector of shape (3 * N_O,)
        """
        # Count waiting operations per operation type
        waiting_ops = np.zeros(self.N_O)
        for job_idx, pos, status in self.operations:
            if status == 'waiting':
                job_type = self.jobs[job_idx][0]
                op_type = self.job_types[job_type][pos]
                waiting_ops[op_type] += 1
        
        # Count idle machines per setup type
        idle_machines = np.zeros(self.N_O)
        for status, setup_type, _, _ in self.machines:
            if status == 'idle':
                idle_machines[setup_type] += 1
        
        # Count in-process operations per operation type
        in_process_ops = np.zeros(self.N_O)
        for job_idx, pos, status in self.operations:
            if status == 'in_process':
                job_type = self.jobs[job_idx][0]
                op_type = self.job_types[job_type][pos]
                in_process_ops[op_type] += 1
        
        # Normalize: waiting_ops by total jobs, others by N_M
        total_jobs = sum(self.production_requirements)
        state = np.concatenate([
            waiting_ops / total_jobs if total_jobs > 0 else waiting_ops,
            idle_machines / self.N_M,
            in_process_ops / self.N_M
        ])
        return state

    def get_possible_actions(self):
        """
        Generate list of possible actions as 4D vectors.
        
        Returns:
            list: Each element is (op_idx, mach_idx, [setup_time, proc_time, rem_ops, rem_time])
        """
        possible_actions = []
        idle_machines = [(i, m) for i, m in enumerate(self.machines) if m[0] == 'idle']
        
        for op_idx, (job_idx, pos, status) in enumerate(self.operations):
            if status != 'waiting':
                continue
            job_type = self.jobs[job_idx][0]
            op_type = self.job_types[job_type][pos]
            
            for mach_idx, (_, setup_type, _, _) in idle_machines:
                if mach_idx not in self.alternative_machines[op_type]:
                    continue
                
                # Compute action features
                setup_time = self.setup_times[setup_type][op_type]
                proc_time = self.processing_times[op_type]
                remaining_ops = len(self.job_types[job_type]) - pos - 1
                remaining_time = sum(
                    self.processing_times[self.job_types[job_type][j]]
                    for j in range(pos + 1, len(self.job_types[job_type]))
                )
                
                # Normalize to [-1, 1]
                features = [
                    2 * (setup_time - self.min_setup_time) / (self.max_setup_time - self.min_setup_time + 1e-6) - 1,
                    2 * (proc_time - self.min_processing_time) / (self.max_processing_time - self.min_processing_time + 1e-6) - 1,
                    2 * (remaining_ops - self.min_remaining_ops) / (self.max_remaining_ops - self.min_remaining_ops + 1e-6) - 1,
                    2 * (remaining_time - self.min_remaining_time) / (self.max_remaining_time - self.min_remaining_time + 1e-6) - 1
                ]
                possible_actions.append((op_idx, mach_idx, features))
        
        return possible_actions

    def step(self, action_index):
        """
        Execute the selected action and update the environment.
        
        Args:
            action_index (int): Index into the list of possible actions
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        possible_actions = self.get_possible_actions()
        if not possible_actions or action_index >= len(possible_actions):
            # No valid actions or invalid action index
            return self.get_state(), 0.0, True, {'error': 'Invalid action'}

        op_idx, mach_idx, _ = possible_actions[action_index]
        job_idx, pos, _ = self.operations[op_idx]
        job_type = self.jobs[job_idx][0]
        op_type = self.job_types[job_type][pos]
        _, current_setup, _, _ = self.machines[mach_idx]

        # Compute setup and processing time
        sigma = self.setup_times[current_setup][op_type]
        proc_time = self.processing_times[op_type]
        finish_time = self.current_time + sigma + proc_time

        # Update machine and operation status
        self.machines[mach_idx] = ('busy', op_type, finish_time, op_idx)
        self.operations[op_idx] = (job_idx, pos, 'in_process')

        # Check if more actions are possible at current time
        next_possible_actions = self.get_possible_actions()
        if next_possible_actions:
            next_time = self.current_time
            reward = -sigma  # Only setup time penalty since time doesn't advance
        else:
            # Advance time to next event (earliest finish time)
            finish_times = [m[2] for m in self.machines if m[0] == 'busy']
            next_time = min(finish_times) if finish_times else self.current_time

            # Update machines and operations that finish
            for i, (status, setup_type, ft, op_id) in enumerate(self.machines):
                if status == 'busy' and ft <= next_time:
                    self.machines[i] = ('idle', setup_type, 0.0, -1)
                    finished_op = self.operations[op_id]
                    self.operations[op_id] = (finished_op[0], finished_op[1], 'completed')
                    # Check if next operation in job can become waiting
                    next_pos = finished_op[1] + 1
                    if next_pos < self.jobs[finished_op[0]][1]:
                        next_op_idx = next(
                            idx for idx, op in enumerate(self.operations)
                            if op[0] == finished_op[0] and op[1] == next_pos
                        )
                        self.operations[next_op_idx] = (finished_op[0], next_pos, 'waiting')

            # Compute idle times for reward
            idle_time_sum = sum(
                (next_time - self.current_time)
                for status, _, ft, _ in self.machines
                if status == 'idle' and ft == 0.0
            )
            reward = -(sigma + idle_time_sum)

        self.current_time = next_time

        # Check if episode is done
        done = all(op[2] == 'completed' for op in self.operations)
        next_state = self.get_state()
        info = {'current_time': self.current_time}

        return next_state, reward, done, info

# Example usage
if __name__ == "__main__":
    # Simplified problem data for testing
    problem_data = {
        'N_M': 2,
        'N_O': 4,
        'N_J': 2,
        'job_types': [[0, 1], [2, 3]],  # J1: O_{0,1}, O_{0,2}; J2: O_{1,1}, O_{1,2}
        'processing_times': [2, 3, 1, 4],  # p_{i,j} for O_{i,j}
        'setup_times': np.array([[0, 1, 2, 1],
                                 [1, 0, 1, 2],
                                 [2, 1, 0, 1],
                                 [1, 2, 1, 0]]),  # sigma_{i',j',i,j}
        'alternative_machines': [[0, 1], [1], [0], [1]],  # E_{i,j}
        'production_requirements': [2, 1]  # P(J_1) = 2, P(J_2) = 1
    }

    env = SemiconductorEnv(problem_data)
    state = env.reset()
    print("Initial state:", state)

    actions = env.get_possible_actions()
    print("Possible actions:", [(op, mach, feat) for op, mach, feat in actions])

    next_state, reward, done, info = env.step(0)
    print("Next state:", next_state)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)