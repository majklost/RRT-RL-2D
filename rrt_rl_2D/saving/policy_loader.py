class PolicyLoader:
    def __init__(self, policy_path, norms_path, additional_args={}):
        self.policy_path = policy_path
        self.norms_path = norms_path
        self.additional_args = additional_args
