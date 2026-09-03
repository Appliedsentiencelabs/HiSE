from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))

from hise.address import StructuralAddress
from hise.rope import MultiDimensionalRoPE


same_text = 'Revenue increased 47% year over year.'
performance = StructuralAddress(18, 2, StructuralAddress.stable_parent_id('Annual Report/Executive Summary'), 1)
risks = StructuralAddress(18, 2, StructuralAddress.stable_parent_id('Annual Report/Risk Factors'), 1)
encoder = MultiDimensionalRoPE(dim=8)
values = __import__('torch').ones(1, 8)
performance_vector = encoder(values, encoder.addresses([performance]))
risks_vector = encoder(values, encoder.addresses([risks]))

print('HiSE structural-address prototype')
print(f'same text: {same_text}')
print(f'performance address: {performance.as_tuple()}')
print(f'risks address:       {risks.as_tuple()}')
print(f'text-only baseline: identical input representation')
print(f'HiSA prototype:      vectors differ = {not __import__("torch").allclose(performance_vector, risks_vector)}')
