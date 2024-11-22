import hashlib

from tools.latency import measure_latency_ms_1
from mmseg.models.backbones.lpsnet import LPSNet

config_file = 'configs/lpsnet/lpsnet_l_90k_4x4b_cityscapes-1536x768.py'

class LPSNetExpansion:
    def __init__(self, depth, width, resolution, parent=None):
        self.depth = depth
        self.width = width
        self.resolution = resolution
        self._check_net_param()
        self.parent = parent
        self._latency = None
        self.miou = -1

        self.delta_depth = ((0, 1, 1, 1, 1), (0, 0, 1, 1, 1), (0, 0, 0, 1, 1))
        self.delta_width = ((4, 8, 16, 32, 32), (0, 8, 16, 32, 32), (0, 0, 16, 32, 32), (0, 0, 0, 32, 32))
        self.delta_resolution = ((1/8, 0, 0), (0, 1/8, 0), (0, 0, 1/8))

    # Ensure that the size of the input parameter size is correct
    # Ensure the element in the parameter are correct
    def _check_net_param(self):
        assert len(self.depth) == 5
        assert len(self.width) == 5
        assert len(self.resolution) == 3
        assert all([d > 0 for d in self.depth])
        assert all([w > 0 for w in self.width])
        assert any([r >= 0 for r in self.resolution])

    @property
    def eval_latency(self):
        if self._latency is None:
            self._latency = measure_latency_ms_1(config_file, self.depth, self.width, self.resolution)
        return self._latency

    def hash(self):
        arch = '{}-{}-{}'.format(self.depth, self.width, self.eval_latency)
        sha1 = hashlib.sha1(arch.encode("utf-8"))
        return sha1.hexdigest()

    def print_arch(self):
        print('Depths:',self.depth)
        print('Channels:',self.width)
        print('Resolution:',self.resolution)

    def _expand_depth(self, op, steplen=1):
        new_depth = [i + j * steplen for i, j in zip(self.depth, op)]
        arch = LPSNetExpansion(depth=new_depth, width=self.width, resolution=self.resolution, parent=self)
        print('    expand_depth: {} + {} * {} = {}, latency={:>0.4f}'.format(
            self.depth, op, steplen, arch.depth, arch.eval_latency))
        return arch

    def _expand_width(self, op, steplen=1):
        new_width = [i + j * steplen for i, j in zip(self.width, op)]
        arch = LPSNetExpansion(depth=self.depth, width=new_width, resolution=self.resolution, parent=self)
        print('    expand_width: {} + {} * {} = {}, latency={:>0.4f}'.format(
            self.width, op, steplen, arch.width, arch.eval_latency))
        return arch

    def _expand_resolution(self, op, steplen=1):
        new_resolution = [i + j * steplen for i, j in zip(self.resolution, op)]
        arch = LPSNetExpansion(depth=self.depth, width=self.width, resolution=new_resolution, parent=self)
        print('    expand_resolution: {} + {} * {} = {}, latency={:>0.4f}'.format(
            self.resolution, op, steplen, arch.resolution, arch.eval_latency))
        return arch

    def _measure_target_latency(self):
        arch_exps = []
        for op in self.delta_depth:
            arch_exps.append(self._expand_depth(op, 1))
        for op in self.delta_width:
            arch_exps.append(self._expand_width(op, 1))
        for op in self.delta_resolution:
            arch_exps.append(self._expand_resolution(op, 1))
        arch_exps_latency = [a._latency for a in arch_exps]
        return max(arch_exps_latency)

    def _expand_depth_to_target(self, target):
        arch_save = []
        for op in self.delta_depth:
            steplen = 1
            while True:
                arch = self._expand_depth(op, steplen)
                if arch.eval_latency >= target:
                    break
                steplen += 1
            # Now arch.latency >= target
            # Also compute arch_prev
            arch_prev = self._expand_depth(op, steplen - 1)
            # Choose the architecture closer to the target latency
            if abs(target - arch_prev.eval_latency) < abs(target - arch.eval_latency):
                arch_save.append(arch_prev)
            else:
                arch_save.append(arch)
        return arch_save

    def _expand_width_to_target(self, target):
        arch_save = []
        for op in self.delta_width:
            steplen = 1
            while True:
                arch = self._expand_width(op, steplen)
                if arch.eval_latency >= target:
                    break
                steplen += 1
            arch_prev = self._expand_width(op, steplen - 1)
            if abs(target - arch_prev.eval_latency) < abs(target - arch.eval_latency):
                arch_save.append(arch_prev)
            else:
                arch_save.append(arch)
        return arch_save

    def _expand_resolution_to_target(self, target):
        arch_save = []
        for op in self.delta_resolution:
            steplen = 1
            while True:
                arch = self._expand_resolution(op, steplen)
                if arch.eval_latency >= target:
                    break
                steplen += 1
            arch_prev = self._expand_resolution(op, steplen - 1)
            if abs(target - arch_prev.eval_latency) < abs(target - arch.eval_latency):
                arch_save.append(arch_prev)
            else:
                arch_save.append(arch)
        return arch_save

    def expand_all(self):
        print('Measuring target latency...')
        target_latency = self._measure_target_latency()
        print('Expanding from latency {:.4f} to {:.4f}'.format(self.eval_latency, target_latency))
        
        expanded_arch = []
        expanded_arch.extend(self._expand_depth_to_target(target_latency))
        expanded_arch.extend(self._expand_width_to_target(target_latency))
        expanded_arch.extend(self._expand_resolution_to_target(target_latency))

        arch_hash = set()
        unique_archs = []
        for arch in expanded_arch:
            if arch.hash not in arch_hash:
                arch_hash.add(arch.hash)
                unique_archs.append(arch)
        return unique_archs

def main():
    depth = [1, 1, 1, 1, 1]
    width = [4, 8, 16, 32, 32]
    resolution = [1/2, 0, 0]

    arch_init = LPSNetExpansion(depth, width, resolution)
    print('Updated latency: ', arch_init.eval_latency)
    
    # Expansion Steps
    best_arch = [arch_init]
    total_steps = 14
    step = 0

    for step in range(step, total_steps):
        print(f'\nExpansion Step {step + 1}')
        base_arch = best_arch[-1]
        print("Last Architecture: ")
        base_arch.print_arch()
        arch_expand = base_arch.expand_all()
        print("Expansion direction: ")
        for arch in arch_expand:
            arch.print_arch()
            print(arch.eval_latency)

        slopes = []
        for i, arch in enumerate(arch_expand, 1):
            print(f'Expanded architecture {i}')
            arch.update_miou()
            print('Updated miou: ', arch.miou)
            print('Parent miou: ', arch.parent.miou)
            print('Updated latency: ', arch.eval_latency)
            print('Parent latency: ', arch.parent.eval_latency)
        
            slope = arch.get_slope()
            print('Slope: ', slope)
            slopes.append(slope)
        
        best_idx = slopes.index(max(slope))
        best_arch.append(arch_expand[best_idx])

if __name__ == '__main__':  
    main()