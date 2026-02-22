"""
Idea pilfered from https://github.com/KellerJordan/Muon
"""

import torch
from torch.optim import AdamW, Muon


class MuonAdamW(torch.optim.Optimizer):
    """
    Wrapper combining Muon (for 2D weight matrices) and AdamW (for
    everything else) into a single optimizer compatible with Lightning
    and PyTorch LR schedulers.
    """
    def __init__(self, muon_params, adam_params, muon_kwargs=None, adam_kwargs=None):
        muon_kwargs = muon_kwargs or {}
        adam_kwargs = adam_kwargs or {}

        self.muon = Muon(muon_params, **muon_kwargs) if muon_params else None
        self.adam = AdamW(adam_params, **adam_kwargs) if adam_params else None

        # Share a single param_groups list across both sub-optimizers so
        # LR schedulers (which store a reference to this list) can modify
        # all groups in-place.
        combined = []
        if self.muon:
            combined.extend(self.muon.param_groups)
        if self.adam:
            combined.extend(self.adam.param_groups)

        self._muon_len = len(self.muon.param_groups) if self.muon else 0

        # Point both sub-optimizers at the shared list so scheduler
        # mutations are visible to both.
        if self.muon:
            self.muon.param_groups = combined[:self._muon_len]
        if self.adam:
            self.adam.param_groups = combined[self._muon_len:]

        # bypass Optimizer.__init__ and just set the attributes directly
        self.defaults = {}
        self.param_groups = combined
        self._optimizer_step_pre_hooks = {}
        self._optimizer_step_post_hooks = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Ensure sub-optimizers see the current param_groups (LR scheduler
        # modifies them in-place on the shared list).
        if self.muon:
            self.muon.param_groups = self.param_groups[:self._muon_len]
        if self.adam:
            self.adam.param_groups = self.param_groups[self._muon_len:]
        if self.muon:
            muon_loss = self.muon.step(closure=None)
            if muon_loss is not None:
                loss = muon_loss
        if self.adam:
            adam_loss = self.adam.step(closure=None)
            if adam_loss is not None:
                loss = adam_loss
        return loss

    def zero_grad(self, set_to_none=True):
        if self.muon:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adam:
            self.adam.zero_grad(set_to_none=set_to_none)

    @property
    def state(self):
        s = {}
        if self.muon:
            s.update(self.muon.state)
        if self.adam:
            s.update(self.adam.state)
        return s

    def state_dict(self):
        return {
            'muon': self.muon.state_dict() if self.muon else None,
            'adam': self.adam.state_dict() if self.adam else None,
        }

    def load_state_dict(self, state_dict):
        if self.muon and state_dict.get('muon'):
            self.muon.load_state_dict(state_dict['muon'])
        if self.adam and state_dict.get('adam'):
            self.adam.load_state_dict(state_dict['adam'])
